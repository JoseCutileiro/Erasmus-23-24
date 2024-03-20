import Data.List
import System.IO

isOdd n
    | mod n 2 == 0 = False
    | otherwise = True

boolToString b
    | b = "True"
    | otherwise = "False"


whatGrade :: Int -> String

whatGrade age
    | (age >= 5) && (age <= 6) = "KinderGarten"
    | (age < 5) = "Lidgi"
    | (age > 6) && (age < 10) = "Elementary school"
    | (age >= 10) && (age <= 18) = "High school"
    | otherwise = "Go to college"

main = do 
    print ("== Guards example ==")
    print ("Is the number 42 odd? " ++ boolToString(isOdd 42))
    print (whatGrade 2)