"""Generate pipelines.

"""

from dsbox.planner.levelone.planner import pipelines_by_hierarchy

if __name__ == "__main__":
    # Generate singleton pipeline by randomly choice a primitve from level 2 of the primitve hierarchy
    pipelines_by_hierarchy(level=2)
