import Data.List
import System.IO

getListItems :: [Int] -> String

getListItems [] = "Your list is empty"
getListItems (x:[]) = "Your list starts with " ++ show x
getListItems (x:y:[]) = "Your list contains " ++ show x ++ " and " ++ show y
getListItems (x:xs) = "The first element of your list is " ++ show x ++ " and the rest is " ++ show xs

getFirstLetter :: String -> String
getFirstLetter [] = "Empty string"
getFirstLetter all@(x:xs) = "The first letter in " ++ all ++ " is " ++ [x]


main = do 

    print(getListItems [])
    print(getListItems [1])
    print(getListItems [1,2])
    print(getListItems [1,2,3])

    print(getFirstLetter "")
    print(getFirstLetter "Hello world")