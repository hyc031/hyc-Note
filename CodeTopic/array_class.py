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

# 题目209 长度最小的子数组
'''
使用快慢指针(滑动窗口)
首先定义两个指针 slow 和 fast，初始都指向数组的起始位置。
然后我们使用一个变量 Sum 来记录当前窗口内的元素和。
判断 Sum 值与target 大小关系, 根据大小关系确定是否要更新 返回的 "最小长度min_len"

# 注意 如果全部未满足应该返回 0  而不是null, 最后多加一条判断即可.

'''
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        slow =0
        fast =0
        Sum =0
        min_len = float('inf')
        while fast <len(nums):
            Sum +=nums[fast]
            while Sum >=target:
                min_len =min(min_len, fast -slow +1)
                Sum -=nums[slow]
                slow +=1
            
            fast +=1
        return min_len if min_len !=float('inf') else 0



# 3 无重复字符的最长子串
'''
给定一个字符串s,请你找出其中不含有重复字符的最长子串的长度。
'''

# tips "集合"  "不重复" --> 想到set()  函数 
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        slow = 0
        fast = 0
        s_set = set()
        max_len = 0

        while fast < len(s):
            if s[fast] in s_set: 
                s_set.remove(s[slow])
                slow += 1
            else:
                s_set.add(s[fast])
                max_len = max(max_len, len(s_set))
                # len(s_set) 可以使用 fast - slow + 1 代替
                fast += 1
        return max_len


# 26 删除有序数组中的重复项

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        slow = 0
        fast = 0
        while fast < len(nums):
            if nums[fast] == nums[slow]:
                fast += 1
            else:
                slow += 1
                nums[slow] = nums[fast]
                fast += 1
        return slow + 1
    














