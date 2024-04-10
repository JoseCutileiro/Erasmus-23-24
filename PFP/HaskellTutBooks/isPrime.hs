isPrime :: Int -> Bool 
isPrime n = n > 1 && all(\x -> n `mod` x/=0) [2..floor (sqrt (fromIntegral n))]

checkMajority :: [Int] -> Bool
checkMajority l = length (filter isPrime l) > fromIntegral (length (l) `div` 2)