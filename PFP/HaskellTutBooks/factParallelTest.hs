import Control.Parallel.Strategies
import System.Random

fact :: Int -> Int 

fact num
    | (num == 0) = 1
    | (num == 1) = 1
    | otherwise = num * (fact $ num -1)


sumFactListSeq :: [Int] -> Int
sumFactListSeq nums = sum $ map fact nums

sumFactListParallel :: [Int] -> Int

sumFactListParallel nums
    | length nums < 100 = sumFactListSeq nums
    | otherwise = sum $ parMap rdeepseq fact nums


main :: IO ()
main = do
    gen <- getStdGen
    let nums = take 8000000 $ randomRs (1, 9) gen :: [Int]
    putStrLn "Sequential Sum:"
    print $ sumFactListSeq nums
    putStrLn "Parallel Sum:"
    print $ sumFactListParallel nums