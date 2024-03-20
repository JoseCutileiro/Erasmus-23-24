import Data.List
import System.IO

num9 = 9 :: Int
num13 = 13 :: Int
-- you have to convert the int
sqrt_9 = sqrt (fromIntegral num9)
sqrt_13=  sqrt (fromIntegral num13)

main = do
    print sqrt_9
    print sqrt_13