sumList :: [Int] -> [Int] -> [Int]
sumList la lb = [a + b | (a,b) <- zip la lb]

isPrime :: Int -> Bool
isPrime n
    | n <= 1    = False
    | otherwise = not $ any (\x -> n `mod` x == 0) [2..intSqrt n]
    where
        intSqrt = floor . sqrt . fromIntegral

primeNum :: Int -> Int
primeNum n
    | isPrime n = n
    | otherwise = 0

compoundFunc :: [Int] -> [Int] -> [Int]
compoundFunc la lb = [(primeNum a) + (primeNum b) | (a,b) <- zip la lb]
 