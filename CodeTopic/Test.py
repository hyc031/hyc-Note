class PrefixSum1D:
    """一维前缀和"""

    def __init__(self, nums: list[int]):
        # prefix[i] = nums[0] + nums[1] + ... + nums[i-1]
        # prefix[0] = 0，方便处理左边界为 0 的情况
        n = len(nums)
        self.prefix = [0] * (n + 1)
        for i in range(n):
            self.prefix[i + 1] = self.prefix[i] + nums[i]
        # [0 1 2 3 4 5 6 7]
        # i in range ()
        # pre[]
        

    def range_sum(self, left: int, right: int) -> int:
        """返回 nums[left..right] 的区间和（闭区间）"""
        return self.prefix[right + 1] - self.prefix[left]


# class PrefixSum2D:
#     """二维前缀和"""

#     def __init__(self, matrix: list[list[int]]):
#         rows = len(matrix)
#         cols = len(matrix[0]) if rows else 0
#         # prefix[i+1][j+1] = matrix 中 (0,0) 到 (i,j) 的子矩阵和
#         self.prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
#         for i in range(rows):
#             for j in range(cols):
#                 self.prefix[i + 1][j + 1] = (
#                     matrix[i][j]
#                     + self.prefix[i][j + 1]
#                     + self.prefix[i + 1][j]
#                     - self.prefix[i][j]
#                 )

#     def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
#         """返回子矩阵 (r1,c1) 到 (r2,c2) 的元素和（闭区间）"""
#         return (
#             self.prefix[r2 + 1][c2 + 1]
#             - self.prefix[r1][c2 + 1]
#             - self.prefix[r2 + 1][c1]
#             + self.prefix[r1][c1]
#         )


# ─── 使用示例 ───────────────────────────────────────────
if __name__ == "__main__":
    # 一维示例
    # nums = [3, 1, 4, 1, 5, 9, 2, 6]
    nums = [0, 1, 2, 3, 4, 5, 6, 7]
    ps1 = PrefixSum1D(nums)
    # print(ps1.range_sum(2, 5))  # nums[2..5] = 4+1+5+9 = 19

    print(ps1.range_sum(1,4)) # nums[1.2.3.4]
    print(len(nums))

    





