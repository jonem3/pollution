# Strategy:

#### Base URL:

http://api.erg.kcl.ac.uk/AirQuality

####Groups URL:
Call: http://api.erg.kcl.ac.uk/AirQuality/Information/Groups/Json
To retrieve all of the locations (groups) for the pollution data

Iterate over: Groups.Group.@GroupName, nothing to save

Iterate over the groups and retrieve pollution data for each group

/Daily/MonitoringIndex/GroupName={GroupName}/Date={Date}/Json

/Daily/MonitoringIndex/Latest/GroupName={GroupName}/Json

####Readings data:
Groups --> Local Authority --> Site
We can store Group & Local Authority but will not use them directly

Data structure will comprise site <-- Pollution observation
cityoflondon
http://api.erg.kcl.ac.uk/AirQuality/Daily/MonitoringIndex/Latest/GroupName=cityoflondon/Json


### Table Definitions
####TABLE pollution_location
* COLUMN site_code <PK>
* COLUMN latitude
* COLUMN longitude

####TABLE pollution_observation
* COLUMN species_code
* COLUMN species_description
* COLUMN air_quality_index
* COLUMN air_quality_band
