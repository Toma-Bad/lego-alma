import operator
#dic of values for button positions
#trying to get the following behaviour
#first button selects the top level value to be changed
#second button usually selects the rough values for the value to be changed (but can also select variable and value!)
#third button controlls modifiers to the rough values, which can be
#adding a smaller value to or multiplying the rough value by a certain amount.

#naming convention: dABc is an array returning values based on some button position
#A = logical position of the first button
#B = 0 - rough value 1 - fine value
#C = used only when the position of the corresponding dAB also selects form multiple variables
#in principle, each array has 4 elements with indices [0,1,2,3] which are selected by the position of a button
#taking the logic values [0,1,2,3]
#example:
#d10 - an arrau of 4 values which are returned
#    - when the first button is on logical position 1 (A=1), 
#    - and represent rough values (B=0) (which are controlled by the second button)

#d11 - an array of 4 values which are returned
#    - when the first button is on logical position 1 (A=1),
#    - represent a modifier (B=1) which in our case are smaller values which will be added to d10
#    - controlled by the third button
#d20 is a special case, as the first two elements of this array correpond to integration time and the last two
#correspond to frequency.
#d210 affects only the d20[0,1] while d211 works with d20[2,3], where the brackets denote indices not values
#the modifier (adding fine values or multiplication) is controlled from var_dic
#in the case of d10 for example, the addition operator (operator.add) is paired to every element of d00,
#meaning in the code the positions of the second and third button returns the sum of corresponding elements
#d00[i] + d01[j]
#in the case of d20, a value from d20[0,1] is multiplied with a value from d210
#while a value from d20[2,3] is added to a value from d211 


#the strings like ['hr_angle'] or ['obj_dec'] help identify which variables is assigned the selected values
#and to understand what each combiunation dABC elements is assigned to
#the [hr] [deg] strings are for display
#the d30 d31 arrays are used for filename selection. Addition of strings concatenates.

#the key values of var dic correspond to positions in the first button.



d00 = [-6,-3,0,+3] #rough values for HA
d01 = [0,1,2,3] #modifiers for HA
d10 = [-80,-50,-30,-10] #rough values for dec
d11 = [0,5,15,20] #modifiers for dec
d20 = [0.2,1,50,500] #rough values for: first two values - int time, last two - frequency
d210 = [1,2,3,12] #modifier for int time
d211 = [0,50,100,250] #modifier for freq
d30 = ['Galaxy','Planet','BH','Misc'] #rough value for file name (first part of filename)
d31 = ['1.jpg','2.jpg','3.jpg','4.jpg'] #modifier for filename (last part of filename)
var_dic = \
	{0:[['hr_angle']*4,list(zip([operator.add]*4,d00)),[d01]*4,["[hr]"]*4], #the d00 d01 values are added
	 1:[['obj_dec']*4,list(zip([operator.add]*4,d10)),[d11]*4,["[deg]"]*4], #the d10 d11 values are added
	 2:[['int_time']*2+ ['obs_freq']*2,list(zip([operator.mul]*2+[operator.add]*2,d20)),[d210]*2+[d211]*2,["[hr]"]*2+["[GHz]"]*2], #explained above, addition or multipl
	 3:[['img_file']*4,list(zip([operator.add]*4,d30)),[d31]*4,[" "]*4]} #string values are concatenated from values in d30 and d31.
	
swi_bits = [52,44,36]

