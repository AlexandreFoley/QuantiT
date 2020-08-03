 /*
 * File: property.h
 * Project: QuanTT
 * File Created: Thursday, 23rd July 2020 11:30:55 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 11:30:55 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

/**
 * 
 * wrapper for properties, allow direct access to users with checks without an explicit setter and getter.
 * id is for the situation where it is desirable to have a different type (setter and getter) for a property with the same owning class.
 * if a completly unique type is necessary an empty lambda []{} can be used for unique_type
*/

template <class content,class owner,class unique_type = owner>
class property final
{
	friend owner;

	content value;
	
	property() = default;
	property(content val):value(val) {}
	public:

	operator const content&() const noexcept {return value;} // read access through implicit conversion

	property& operator=( content new_value ); // define it to give write access to the value, with any and all checks necessary.

	~property() {};
};