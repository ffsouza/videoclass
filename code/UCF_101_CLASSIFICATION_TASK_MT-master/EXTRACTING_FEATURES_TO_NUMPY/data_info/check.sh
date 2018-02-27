cat trainPart0.txt > train_DELETE.txt 
cat trainPart1.txt >> train_DELETE.txt 
cat trainPart2.txt >> train_DELETE.txt 
cat trainPart3.txt >> train_DELETE.txt 
cat trainPart4.txt >> train_DELETE.txt 
cat trainPart5.txt >> train_DELETE.txt 
cat trainPart6.txt >> train_DELETE.txt 
cat trainPart7.txt >> train_DELETE.txt 
cat trainPart8.txt >> train_DELETE.txt 
cat trainPart9.txt >> train_DELETE.txt 

diff train_DELETE.txt trainlist01.txt
cat testPart0.txt > test_DELETE.txt 
cat testPart1.txt >> test_DELETE.txt 
cat testPart2.txt >> test_DELETE.txt 
cat testPart3.txt >> test_DELETE.txt 
cat testPart4.txt >> test_DELETE.txt 
cat testPart5.txt >> test_DELETE.txt 
cat testPart6.txt >> test_DELETE.txt 
cat testPart7.txt >> test_DELETE.txt 
cat testPart8.txt >> test_DELETE.txt 
cat testPart9.txt >> test_DELETE.txt 

diff test_DELETE.txt testlist01.txt
