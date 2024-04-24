def sumList (arr: []i32) : i32= 
 reduce (+) 0 arr

def mult2 (arr: []i32) : []i32 = 
    map (\x -> x * 2) arr

def ex_1 (size : i64) : i32 =
    let arr = iota size
    let cast = map (\x -> i32.i64 x) arr
    let arr_2 = mult2 cast
    in sumList arr_2
