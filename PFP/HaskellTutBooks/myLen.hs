myLen :: [a] -> Int
myLen [] = 0
myLen (x:xs) = 1 + myLen xs

sumLen :: [[a]] -> Int
sumLen [] = 0
sumLen (x:xs) = myLen x + sumLen xs