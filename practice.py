class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_index=0
        max_len_index=0
        max_len=0
        left=0
        for right in range(len(s)):
            c=s[right]
            if c in char_index and char_index[c]>=left:
                left=char_index[c]+1
            char_index[c]=right
            max_len=max(max_len,right-left+1)
        return max_len
result=Solution()
print(result.lengthOfLongestSubstring("abcabcbb"))