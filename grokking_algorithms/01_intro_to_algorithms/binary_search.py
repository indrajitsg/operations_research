"""Simple function to perform binary search"""

def binary_search(arr, item):
    """Use binary search to find position of an item in an array"""
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]
        if guess == item:
            return mid
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None


if __name__ == "__main__":
    my_list = [1, 3, 5, 7, 9]
    item = 7
    print(f"Checking if item {item} is present")
    print(f"Answer: {binary_search(my_list, item)}")
