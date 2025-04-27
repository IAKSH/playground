import sys
def inverse_num(array):
    n = len(array)
    if n <= 1:
        return 0, array

    mid = n // 2
    # 拆分为两个部分然后向下递归
    inverse_l, arr_l = inverse_num(array[:mid])
    inverse_r, arr_r = inverse_num(array[mid:])

    nl, nr = len(arr_l), len(arr_r)

    arr_l.append(sys.maxsize)
    arr_r.append(sys.maxsize)

    i, j = 0, 0
    new_arr = []
    inverse = inverse_l + inverse_r

    while i < nl or j < nr:
        if arr_l[i] <= arr_r[j]:
            inverse += j
            new_arr.append(arr_l[i])
            i += 1
        else:
            new_arr.append(arr_r[j])
            j += 1
    return inverse, new_arr

print(inverse_num([1,3,5,7,8,6,4,2]))