
access :: Int -> [Int] -> Int
access n l = head $ drop n l

edit :: Int -> Int -> [Int] -> [Int]

edit index new_value l = index `take` l ++ [new_value] ++ (index + 1) `drop` l


main = do 
     print $ access 1 [1,2,3,4,5,6]
     print $ access 2 [1,2,3,4,5,6]
     print $ access 3 [1,2,3,4,5,6]
     print $ access 4 [1,2,3,4,5,6]
     print $ edit 0 10 [1,2,3,4,5,6]



























