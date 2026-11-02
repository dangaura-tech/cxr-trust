"""Initialise a repo and create a sequence of logical commits.
usage: python commit_plan.py <repo_dir> <plan.txt>
plan format: one commit per block:  message line, then path lines, blank line between commits.
Paths not listed are committed in a final 'chore' commit if any remain."""
import subprocess, sys, os
repo, plan = sys.argv[1], sys.argv[2]
def git(*a): subprocess.run(["git", "-C", repo, *a], check=True, stdout=subprocess.DEVNULL)
env_ok = True
if not os.path.isdir(os.path.join(repo, ".git")):
    git("init", "-q", "-b", "main")
git("config", "user.name", "dangaura-tech")
git("config", "user.email", "aurenlucian.letmdang@gmail.com")
git("config", "commit.gpgsign", "false")
blocks = [b.strip().splitlines() for b in open(plan).read().split("\n\n") if b.strip()]
for b in blocks:
    msg, paths = b[0], b[1:]
    git("add", "--", *paths)
    git("commit", "-q", "-m", msg)
rest = subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True).stdout
if rest.strip():
    print("UNCOMMITTED:\n" + rest)
print(subprocess.run(["git", "-C", repo, "log", "--oneline"], capture_output=True, text=True).stdout)
