"""Generate validated PR evidence plus an author explanation; defaults to english-v1."""

from s1mb.submission_report import main

if __name__ == "__main__":
    main(template=True)
