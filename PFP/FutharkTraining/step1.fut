-- Author: José Cutileiro
-- Source: https://futhark-lang.org/examples.html


-- Factorial

def fact (n: i32): i32 = reduce (*) 1 (1...n)

-- Types 

def an_int : i32 = 2
def an_unsigned_byte : u8 = 2
def a_bool = true
def a_double : f64 = 2.0

-- Type conversion

let x : i64 = 2
let y : f32 = f32.i64 x

-- Math function examples

def hypo (x: f32) (y: f32) = f32.sqrt (x*x + y*y)

-- Functions

def plus2 (x : i32) : i32 = 
    x + 2

-- Function as argument

def on_i32 (f: i32 -> i32) (x: i32) = 
    f x

-- Anonymous function (constant value four)

def four = on_i32 (\x -> x + 2) 2

-- Arrays

def first (arr : []i32) : i32 =
    arr[0]

def reverse (arr: []i32) : []i32 = 
    arr[::-1]

-- map (futhark compiler know how to  convert map into parallel code)

def sum2List (xs: []i32) : []i32= map (+2) xs
def mult3List (xs : []i32) : []i32 = map (*3) xs

-- zip (combine arrays in pairs)

def pairs xs ys = zip xs ys

def sumLists xs ys = map(\(x,y) -> x + y) (pairs xs ys)

-- combine zip and map -> map 2

def sumLists_map2 xs ys = map2 (\x y -> x + y) xs ys

-- with  type definition 
-- (because without is assumed as int)
-- this removes the warning too 

def sumLists_Float (xs: []f32) (ys: []f32) : []f32 =
    map2 (\x y -> x + y) xs ys


-- tuples

def a_tuple : (i32, bool) = (1, true)

def projection_1 = a_tuple.0          -- 1
def projection_2 = a_tuple.1          -- true

-- append

def append xs ys = xs ++ ys

-- range

def ints = 1...10                   -- [1,2 ... ,10]


