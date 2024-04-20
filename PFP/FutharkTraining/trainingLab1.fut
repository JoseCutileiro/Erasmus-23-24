-- absolute value
def abs = \(x : i32) ->
    if x < 0
    then -x
    else x

-- get max element in a list
def getMax (xs : []i32) : i32 = 
    reduce 
        (\max x -> if max > x then max else x) 0 xs

-- max element (in the documentation)
def mx (m1:i32,i1:i64) (m2:i32,i2:i64) : (i32,i64) =
  if m1 > m2 then (m1,i1) else (m2,i2)

-- max index (in the documentation)
def maxidx [n] (xs: [n]i32) : (i32,i64) =
  reduce mx (i32.lowest,-1) (zip xs (iota n))