# Hot 100 中的一些题目
# 01 两数之和
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
        # cache 中 key: i(下表0 1 2 3 )  value: nums (nums[0], nums[1]...)
        for i, item in enumerate(nums):
            other = target - item
            if other in cache and cache[other] != i:
                return [i, cache[other]]
 

# 15 三数之和
'''
题目中没有明确说明 三元组的顺序
i < j < k
答案中不能重复出现 三元组  nums[-1, 0, 1, 2, -1, -4]  不能出现两个[-1, 0, 1]
'''

class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        ans = []
        n = len(nums)
        for i in range(n-2):
            x = nums[i]
            if i > 0 and x == nums[i-1]:
                continue
            # if x + nums[i+1] + nums[i+2] > 0: 优化1 
            #     break
            # if x + nums[-2] + nums[-1] < 0: 优化2
            #     continue
            j = i + 1
            k = n - 1
            while j < k:
                s = x + nums[j] + nums[k]
                if s > 0:
                    k -= 1
                elif s < 0:
                    j += 1
                else:
                    ans.append([x, nums[j], nums[k]])
                    j += 1
                    while j <k and nums[j] == nums[j-1]:
                        j += 1
                    k -= 1
                    while k> j and nums[k] == nums[k+1]:
                        k -= 1
        return ans 

num = [-1, 0, 1, 2, -1, -4]
solution = Solution()
ans = solution.threeSum(num)
print(ans)








