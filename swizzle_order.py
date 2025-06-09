from typing import List, Tuple


def generate_cta_schedule(
    block_idx,
    shape_m: int,
    shape_n: int,
    block_m: int,
    block_n: int,
    num_1d_blocks_per_group: int,
    multicast_on: str = "A",
) -> List[List[Tuple[int, int]]]:
    """
    Returns a list of groups. Each group is a list of (m_block_idx, n_block_idx) tuples.

    multicast_on: either "A" or "B"
        - "A" means multicast is along M axis → N is primary
        - "B" means multicast is along N axis → M is primary
    """
    assert multicast_on in ("A", "B")

    num_m_blocks = (shape_m + block_m - 1) // block_m
    num_n_blocks = (shape_n + block_n - 1) // block_n

    if multicast_on == "A":
        # Primary = N, Secondary = M (row-major over (m, n))
        primary_blocks = num_n_blocks
        secondary_blocks = num_m_blocks
    else:
        # Primary = M, Secondary = N (col-major over (m, n))
        primary_blocks = num_m_blocks
        secondary_blocks = num_n_blocks

    schedule = []
    num_blocks_per_group = secondary_blocks * num_1d_blocks_per_group
    group_idx = block_idx // num_blocks_per_group
    first_block_idx = group_idx * num_1d_blocks_per_group
    in_group_idx = block_idx % num_blocks_per_group
    num_blocks_in_group = min(num_1d_blocks_per_group, primary_blocks - first_block_idx)

    if multicast_on == "A":
        m_block_idx = in_group_idx // num_blocks_in_group
        n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group
    else:
        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group
        n_block_idx = in_group_idx // num_blocks_in_group

    return group_idx, m_block_idx, n_block_idx

# Experimenting with multicast on A (CUTLASS default)
schedule_a = [generate_cta_schedule(i,
    shape_m=128,
    shape_n=128,
    block_m=32,
    block_n=32,
    num_1d_blocks_per_group=2,
    multicast_on="A",
) for i in range(8)]

# # Multicast on B (your alternative layout)
# schedule_b = generate_cta_schedule(
#     shape_m=128,
#     shape_n=128,
#     block_m=32,
#     block_n=32,
#     num_1d_blocks_per_group=2,
#     multicast_on="B",
# )

# Print out group-wise traversal
from itertools import groupby

groups = {g:[(m,n) for g,m,n in idx] for g,idx in groupby(schedule_a, key=lambda x: x[0])}

for group, block_idx in groups.items():
    print(f"Group {group} (multicast A): {list(block_idx)}")
