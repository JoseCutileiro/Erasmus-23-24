
myDrop :: Int -> [Int] -> [Int]
myDrop n l
	| n <= 0 || [] == l = l
	| otherwise = myDrop (n-1) (tail l)

main = do
	print (myDrop 3 [1,2,3,4,5])
