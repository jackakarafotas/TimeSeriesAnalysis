#ifndef __PARSER_H
#define __PARSER_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
using namespace std;

class Parser {
public:
	// Constructor
	Parser();

	// Virtual Destructor
	virtual ~Parser();

	// Main function
	void parse_data(
		const string& file_location,
		vector<string>& dates,
		vector<double>& price_open,
		vector<double>& price_close,
		vector<double>& price_high,
		vector<double>& price_low,
		vector<double>& volume) const;
};

Parser::Parser() {}

Parser::~Parser() {}

void Parser::parse_data(
	const string& file_location,
	vector<string>& date_series,
	vector<double>& price_open_series,
	vector<double>& price_close_series,
	vector<double>& price_high_series,
	vector<double>& price_low_series,
	vector<double>& volume_series) const {
	/* 
	- Parses CSVs downloaded from yahoo finance
	- Stores data in vectors of doubles
	*/ 

	ifstream file(file_location);
	string line;
	string adj_close;

	string date;
	string price_open;
	string price_close;
	string price_high;
	string price_low;
	string volume;

	getline(file,line);

	if (file.is_open()) {
		while (getline(file,date,',')) {
			date_series.push_back(date);

			getline(file,price_open,',');
			price_open_series.push_back(stod(price_open));

			getline(file,price_high,',');
			price_high_series.push_back(stod(price_high));

			getline(file,price_low,',');
			price_low_series.push_back(stod(price_low));

			getline(file,price_close,',');
			price_close_series.push_back(stod(price_close));

			getline(file,adj_close,',');

			getline(file,volume);
			volume_series.push_back(stod(volume));
		}
	} else {
		cout << "Unable to open file or Ticker does not exist. Exiting.";
	}

}















#endif