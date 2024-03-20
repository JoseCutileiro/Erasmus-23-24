import Data.List
import System.IO

-- Generate even list until 20
evenList = [2,4..20]

-- Generate chars
letterList = ['A'..'Z']

-- jump one letter
jumpOne = ['A','C'..'Z']

-- take 10 two's
many2s = take 10 (repeat 2)

-- replicate 10 three's
many3s = replicate 10 3

-- cycle
cycleAList = take 13 (cycle [1,2,3,4,5])

main = do
    print evenList
    print letterList
    print jumpOne
    print many2s
    print many3s
    print cycleAList