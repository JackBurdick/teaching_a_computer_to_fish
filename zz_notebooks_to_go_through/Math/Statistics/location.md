# Terms

## Estimates of Location

- Mean (average)
> The sum of all values divided by the total number of values
- Weighted mean (weighted average)
> The sum of (all values multiplied by a weight) divided by the total number of values.
- Trimmed mean (truncated mean)
> The mean of all values after removing a fixed number of extreme values.  This is used to eliminate the influence of the removed extreme values.  -- A common fixed value may be ~10%; thus, the top and bottom 10% of the data may be removed.
- Median
> The value in the 'middle' of the dataset - such that half of the data lies above the point and half of the data lies below the point. NOTE: if the data length is even, then the median is actually not in the dataset; rather, the median is the mean of the two center most values.
- Weighted Median
> The value in the 'middle' of the weighted dataset.  To calculate the weighted median, the data is first sorted, then instead of finding the middle value - the weighted median is the value such taht the sum o fthe weights is equal above and below the sorted list.
- Outlier (extreme value)
> In general terms, an outlier is a value in the data that is exceptionally different from most of the data

## Other terms

- Robust (resistant)
> Robust indicates that something is not sensitive to extreme values

## Notes

Why might a 'metric' be weighted?
> sometimes a some values are more 'important' than others. two examples:
> - If collecting data from an array of sensors and certain sensors are known to be more accurate than others, then we may assign a higher weight to the "better" sensor data.  This is in contrast to keeping all data (which may be affected by the poor sensors, and|or removing the data from the sensors that are known to be "worse")
> - If collecting data from a variety of data sources, some methods of collecting data may not be representative of the whole population as much as others and so each data source may be weighted to correct for this.


## Resources
NOTE: Definitions are largely adapted from resources [1]
- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.amazon.com/Practical-Statistics-Data-Scientists-Essential/dp/1491952962/ref=sr_1_1?ie=UTF8&qid=1515452947&sr=8-1&keywords=practical+statistics+for+data+scientists)