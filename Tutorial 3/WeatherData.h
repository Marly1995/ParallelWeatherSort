#pragma once
#include <iostream>
#include <vector>

class WeatherData
{
public:
	std::string station;
	int year;
	int month;
	int day;
	int time;
	int temperature;

public:
	WeatherData();
	~WeatherData();
};