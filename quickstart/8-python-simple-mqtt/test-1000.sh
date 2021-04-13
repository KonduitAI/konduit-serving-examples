for (( c=0; c<=$1; c++ ))
do 
   data='{"a":'$c',"b":'$c'}'
   echo "Sending $data"   
   konduit predict python-mqtt $data
done
