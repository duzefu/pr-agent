from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import gitlab
from gitlab import GitlabGetError

from pr_agent.algo.file_filter import filter_ignored
from pr_agent.algo.git_patch_processing import decode_if_bytes
from pr_agent.algo.language_handler import is_valid_file
from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo
from pr_agent.algo.utils import clip_tokens, load_large_diff
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.git_provider import MAX_FILES_ALLOWED_FULL, GitProvider
from pr_agent.log import get_logger


@dataclass
class _PushPR:
    """Minimal PR-like object for push events (no Merge Request required)."""
    title: str
    description: str
    source_branch: str
    draft: bool = False
    labels: list = field(default_factory=list)


class GitLabPushProvider(GitProvider):
    """
    GitProvider implementation for GitLab push events.

    Handles `object_kind == 'push'` webhook events without requiring a Merge Request.
    Uses GitLab's repository_compare API to obtain diffs and posts review comments
    directly to the push's head commit.

    Push URL format (internal identifier):
        {project_url}/-/push/{before_sha}/{after_sha}
    """

    def __init__(self, push_url: str, incremental: bool = False):
        gitlab_url = get_settings().get("GITLAB.URL", None)
        if not gitlab_url:
            raise ValueError("GitLab URL is not set in the config file")
        self.gitlab_url = gitlab_url

        ssl_verify = get_settings().get("GITLAB.SSL_VERIFY", True)
        gitlab_access_token = get_settings().get("GITLAB.PERSONAL_ACCESS_TOKEN", None)
        if not gitlab_access_token:
            raise ValueError("GitLab personal access token is not set in the config file")

        auth_method = get_settings().get("GITLAB.AUTH_TYPE", "oauth_token")
        if auth_method not in ["oauth_token", "private_token"]:
            raise ValueError(f"Unsupported GITLAB.AUTH_TYPE: '{auth_method}'. Must be 'oauth_token' or 'private_token'.")

        if auth_method == "oauth_token":
            self.gl = gitlab.Gitlab(url=gitlab_url, oauth_token=gitlab_access_token, ssl_verify=ssl_verify)
        else:
            self.gl = gitlab.Gitlab(url=gitlab_url, private_token=gitlab_access_token, ssl_verify=ssl_verify)

        self.pr_url = push_url
        self.max_comment_chars = 65000
        self.diff_files = None
        self.temp_comments = []
        self.incremental = incremental

        self._parse_push_url(push_url)
        self.project = self.gl.projects.get(self.project_path)

        # Read push context from starlette context (set by webhook handler)
        self.branch = ''
        self.push_commits = []
        try:
            from starlette_context import context
            self.branch = context.get('push_branch', '') or ''
            self.push_commits = context.get('push_commits', []) or []
        except Exception:
            pass

        self._pr_obj = self._build_pr_obj()

    def _parse_push_url(self, push_url: str):
        """
        Parse push URL: {project_url}/-/push/{before_sha}/{after_sha}
        """
        try:
            marker = '/-/push/'
            idx = push_url.index(marker)
            project_url = push_url[:idx]
            shas_part = push_url[idx + len(marker):]
            parts = shas_part.split('/', 1)
            self.before_sha = parts[0]
            self.after_sha = parts[1] if len(parts) > 1 else parts[0]
            parsed = urlparse(project_url)
            self.project_path = parsed.path.strip('/')
        except Exception as e:
            raise ValueError(f"Invalid push URL format: {push_url}") from e

    def _build_pr_obj(self) -> _PushPR:
        last_commit_msg = ''
        if self.push_commits:
            last_commit_msg = self.push_commits[-1].get('message', '').split('\n')[0].strip()
        if not last_commit_msg:
            try:
                commit = self.project.commits.get(self.after_sha)
                last_commit_msg = (commit.title or commit.message.split('\n')[0]).strip()
            except Exception:
                last_commit_msg = self.after_sha[:8]

        title = f"Push to {self.branch}: {last_commit_msg}" if self.branch else f"Push: {last_commit_msg}"
        description = self._build_description()
        return _PushPR(title=title, description=description, source_branch=self.branch)

    def _build_description(self) -> str:
        if not self.push_commits:
            return f"Push event: {self.before_sha[:8]}..{self.after_sha[:8]}"
        lines = [f"{i + 1}. {c.get('message', '').strip()}" for i, c in enumerate(self.push_commits)]
        return "\n".join(lines)

    @property
    def pr(self) -> _PushPR:
        """Return a PR-like object for use by PR Agent tools."""
        return self._pr_obj

    def is_supported(self, capability: str) -> bool:
        unsupported = {
            'create_inline_comment',
            'publish_inline_comments',
            'publish_file_comments',
            'gfm_markdown',
            'get_issue_comments',
        }
        return capability not in unsupported

    # ---- diff / file methods ----

    def get_files(self) -> list:
        return [f.filename for f in self.get_diff_files()]

    def get_diff_files(self) -> list[FilePatchInfo]:
        if self.diff_files:
            return self.diff_files

        try:
            cmp = self.project.repository_compare(self.before_sha, self.after_sha)
        except Exception as e:
            get_logger().error(f"GitLabPushProvider: failed to compare {self.before_sha}..{self.after_sha}: {e}")
            return []

        if isinstance(cmp, dict):
            raw_changes = cmp.get('diffs', [])
        else:
            raw_changes = getattr(cmp, 'diffs', []) or []

        diffs_original = raw_changes
        diffs = filter_ignored(diffs_original, 'gitlab')
        if diffs != diffs_original:
            try:
                names_orig = [d.get('new_path', '') if isinstance(d, dict) else getattr(d, 'new_path', '') for d in diffs_original]
                names_filt = [d.get('new_path', '') if isinstance(d, dict) else getattr(d, 'new_path', '') for d in diffs]
                get_logger().info("Filtered out [ignore] files for push", extra={
                    'original_files': names_orig,
                    'filtered_files': names_filt,
                })
            except Exception:
                pass

        diff_files = []
        invalid_files = []
        counter_valid = 0

        for diff in diffs:
            if isinstance(diff, dict):
                new_path = diff.get('new_path', '')
                old_path = diff.get('old_path', new_path)
                patch = diff.get('diff', '')
                is_new = diff.get('new_file', False)
                is_deleted = diff.get('deleted_file', False)
                is_renamed = diff.get('renamed_file', False)
            else:
                new_path = getattr(diff, 'new_path', '')
                old_path = getattr(diff, 'old_path', new_path)
                patch = getattr(diff, 'diff', '')
                is_new = getattr(diff, 'new_file', False)
                is_deleted = getattr(diff, 'deleted_file', False)
                is_renamed = getattr(diff, 'renamed_file', False)

            if not is_valid_file(new_path):
                invalid_files.append(new_path)
                continue

            counter_valid += 1
            if counter_valid < MAX_FILES_ALLOWED_FULL or not patch:
                original_content = self._get_file_content(old_path, self.before_sha)
                new_content = self._get_file_content(new_path, self.after_sha)
            else:
                if counter_valid == MAX_FILES_ALLOWED_FULL:
                    get_logger().info("Too many files in push diff, skipping full content for remaining files")
                original_content = ''
                new_content = ''

            original_content = decode_if_bytes(original_content)
            new_content = decode_if_bytes(new_content)

            if is_new:
                edit_type = EDIT_TYPE.ADDED
            elif is_deleted:
                edit_type = EDIT_TYPE.DELETED
            elif is_renamed:
                edit_type = EDIT_TYPE.RENAMED
            else:
                edit_type = EDIT_TYPE.MODIFIED

            filename = new_path
            if not patch:
                patch = load_large_diff(filename, new_content, original_content)

            patch_lines = patch.splitlines(keepends=True) if patch else []
            num_plus = len([ln for ln in patch_lines if ln.startswith('+')])
            num_minus = len([ln for ln in patch_lines if ln.startswith('-')])

            diff_files.append(FilePatchInfo(
                original_content, new_content,
                patch=patch,
                filename=filename,
                edit_type=edit_type,
                old_filename=None if old_path == new_path else old_path,
                num_plus_lines=num_plus,
                num_minus_lines=num_minus,
            ))

        if invalid_files:
            get_logger().info(f"Filtered out files with invalid extensions: {invalid_files}")

        self.diff_files = diff_files
        return diff_files

    def _get_file_content(self, file_path: str, ref: str) -> str:
        if not file_path or not ref:
            return ''
        try:
            f = self.project.files.get(file_path=file_path, ref=ref)
            content = f.decode()
            return decode_if_bytes(content)
        except GitlabGetError:
            return ''
        except Exception as e:
            get_logger().warning(f"GitLabPushProvider: error retrieving {file_path}@{ref}: {e}")
            return ''

    # ---- comment / output methods ----

    def publish_description(self, pr_title: str, pr_body: str):
        pass  # Commits have no editable description

    def publish_comment(self, comment: str, is_temporary: bool = False):
        if is_temporary and not get_settings().config.publish_output_progress:
            get_logger().debug(f"Skipping temporary comment (progress disabled)")
            return None
        comment = self.limit_output_characters(comment, self.max_comment_chars)
        try:
            commit = self.project.commits.get(self.after_sha)
            note = commit.comments.create({'note': comment})
            if is_temporary:
                self.temp_comments.append(note)
            return note
        except Exception as e:
            get_logger().error(f"GitLabPushProvider: failed to publish commit comment: {e}")
            return None

    def publish_persistent_comment(self, pr_comment: str,
                                   initial_header: str,
                                   update_header: bool = True,
                                   name='review',
                                   final_update_message=True):
        # GitLab commit comments cannot be edited, always create a new comment
        self.publish_comment(pr_comment)

    def remove_initial_comment(self):
        # GitLab commit comments cannot be deleted via API; just clear the list
        self.temp_comments = []

    def remove_comment(self, comment):
        pass  # Commit comments cannot be deleted via the GitLab API

    def get_issue_comments(self):
        try:
            commit = self.project.commits.get(self.after_sha)
            return commit.comments.list(get_all=True)
        except Exception as e:
            get_logger().error(f"GitLabPushProvider: failed to get commit comments: {e}")
            return []

    # ---- PR metadata methods ----

    def get_pr_description_full(self) -> str:
        return self._pr_obj.description

    def get_pr_branch(self) -> str:
        return self.branch

    def get_commit_messages(self) -> str:
        max_tokens = get_settings().get("CONFIG.MAX_COMMITS_TOKENS", None)
        try:
            messages = [c.get('message', '').strip() for c in self.push_commits]
            result = "\n".join([f"{i + 1}. {m}" for i, m in enumerate(messages)])
        except Exception:
            result = ''
        if max_tokens:
            result = clip_tokens(result, max_tokens)
        return result

    def get_languages(self) -> dict:
        try:
            return self.project.languages()
        except Exception as e:
            get_logger().warning(f"GitLabPushProvider: failed to get languages: {e}")
            return {}

    def get_repo_settings(self):
        try:
            contents = self.project.files.get(file_path='.pr_agent.toml', ref=self.after_sha).decode()
            if isinstance(contents, str):
                return contents.encode()
            return contents
        except Exception:
            return b''

    def get_pr_id(self) -> str:
        return self.pr_url or ''

    def get_latest_commit_url(self) -> str:
        try:
            commit = self.project.commits.get(self.after_sha)
            return getattr(commit, 'web_url', '')
        except Exception:
            return ''

    # ---- no-op / stub methods for MR-specific capabilities ----

    def get_user_id(self):
        return None

    def publish_code_suggestions(self, code_suggestions: list) -> bool:
        return False

    def publish_inline_comment(self, body: str, relevant_file: str, relevant_line_in_file: str,
                               original_suggestion=None):
        pass

    def publish_inline_comments(self, comments: list[dict]):
        pass

    def publish_labels(self, labels):
        pass

    def get_pr_labels(self, update=False):
        return []

    def add_eyes_reaction(self, issue_comment_id: int, disable_eyes: bool = False) -> Optional[int]:
        return None

    def remove_reaction(self, issue_comment_id: int, reaction_id: int) -> bool:
        return False
