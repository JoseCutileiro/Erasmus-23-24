qsort ::  [Int] -> [Int]
qsort [] = []
qsort (x:xs) = qsort smaller ++ [x] ++ qsort larger
    where
        smaller = filter (< x) xs 
        larger = filter (>= x) xs