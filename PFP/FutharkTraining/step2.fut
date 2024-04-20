-- Author: José Cutileiro
-- Source: https://futhark-lang.org/examples.html

-- reduce

def average (xs: []f32) : f32 = 
    reduce (+) 0.0 xs / f32.i64 (length xs)

-- scan <=> apply reduce to:
--      [1]
--      [1,2]
--      [1,2,3]
--      [1,2,3,4]

def testing_scan (xs: []i32) : []i32 =
    scan (+) 0 xs

-- PROXIMO -> gather and scatter