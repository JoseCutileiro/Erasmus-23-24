import Data.List
import System.IO

classAvg :: Double -> Double -> String 

classAvg grade_test1 grade_test2
    | avg <= 9.5 = "Failed"
    | avg <= 14.5 = "Passed"
    | avg <= 17.5 = "very good"
    | avg <= 20 = "Excellent"
    | otherwise = "Invalid input inserted"
    where avg = (grade_test1 + grade_test2) / 2.0

main = do 
    print (classAvg 2 5)