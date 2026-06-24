nums = [3,4,5,10]
cache = {}
for i, item in enumerate(nums):
    cache[item] = i


print(cache)
print(cache[10])

