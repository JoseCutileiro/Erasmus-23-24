import Control.Parallel (par, pseq)

merge_sort [] = []
merge_sort [x] = [x]
merge_sort xs = merge (merge_sort ys) (merge_sort zs) 
    where (ys, zs) = split xs

merge [] ys = ys
merge xs [] = xs 
merge (x:xs) (y:ys) | x <= y      = x:merge xs (y:ys)
                   | otherwise    = y:merge (x:xs) ys

split xs = split2 [] xs xs

split2 xs (y:ys) (_:_:zs) = split2 (y:xs) ys zs 
split2 xs ys _            = (xs,ys)

merge_sort_par [] = []
merge_sort_par [x] = [x]
merge_sort_par xs
    | length xs < 1000 = merge_sort xs 
    | otherwise = sortedYs `par` (sortedZs `pseq` merge sortedYs sortedZs)
    where 
        (ys,zs) = split xs
        sortedYs = merge_sort_par ys
        sortedZs = merge_sort_par zs 

