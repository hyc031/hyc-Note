
class Solution:
    # 遍历求解
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        for i, val in enumerate(nums):
            for j in range(i+1, len(nums)):
                ans = nums[j]
                if val + ans == target:
                    return [i, j]
class Solution2:
    # 使用哈希表求解
    def twoSum2(self, nums: list[int], target: int) -> list[int]:
        cache = {}
        for i, item in enumerate(nums):
            cache[item] = i
        # cache = {key: value, key2: value2, ....}
        for i, item in enumerate(nums):
            other = target - item
            if other in cache and cache[other] != i:
                return [i, cache[other]]






nums_1 = [1,2,3,6]
tag = 5

if __name__ == '__main__':
    s = Solution()
    ans = s.twoSum(nums_1, tag)
    print(ans) 








