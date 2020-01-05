[Home](../README.md) | Datapoint API
# Description of the datapoint API

##### Example to get a list of sites:
[http://datapoint.metoffice.gov.uk/public/data/val/wxobs/all/json/sitelist?key=2535722c-4f5e-4b85-bece-94d35f534658](http://datapoint.metoffice.gov.uk/public/data/val/wxobs/all/json/sitelist?key=2535722c-4f5e-4b85-bece-94d35f534658)

Example of 1 line from the above:
```json
{
	"elevation": "25.0",
	"id": "3772",
	"latitude": "51.479",
	"longitude": "-0.449",
	"name": "Heathrow",
	"region": "se",
	"unitaryAuthArea": "Greater London"
}
```

Once you have the list of sites you can filter based upon whatever criteria you want.

Having filtered the locations the ID is used to query data for that location
as in the following example:

[http://datapoint.metoffice.gov.uk/public/data/val/wxobs/all/json/3772?res=hourly&key=2535722c-4f5e-4b85-bece-94d35f534658](http://datapoint.metoffice.gov.uk/public/data/val/wxobs/all/json/3772?res=hourly&key=2535722c-4f5e-4b85-bece-94d35f534658)

```json
{
	"SiteRep": {
		"Wx": {
			"Param": [{
				"name": "G",
				"units": "mph",
				"$": "Wind Gust"
			}, {
				"name": "T",
				"units": "C",
				"$": "Temperature"
			}, {
				"name": "V",
				"units": "m",
				"$": "Visibility"
			}, {
				"name": "D",
				"units": "compass",
				"$": "Wind Direction"
			}, {
				"name": "S",
				"units": "mph",
				"$": "Wind Speed"
			}, {
				"name": "W",
				"units": "",
				"$": "Weather Type"
			}, {
				"name": "P",
				"units": "hpa",
				"$": "Pressure"
			}, {
				"name": "Pt",
				"units": "Pa\/s",
				"$": "Pressure Tendency"
			}, {
				"name": "Dp",
				"units": "C",
				"$": "Dew Point"
			}, {
				"name": "H",
				"units": "%",
				"$": "Screen Relative Humidity"
			}]
		},
		"DV": {
			"dataDate": "2020-01-05T12:00:00Z",
			"type": "Obs",
			"Location": {
				"i": "3772",
				"lat": "51.479",
				"lon": "-0.449",
				"name": "HEATHROW",
				"country": "ENGLAND",
				"continent": "EUROPE",
				"elevation": "25.0",
"Period":[
{"type":"Day","value":"2020-01-04Z","Rep":[
  {"D":"W","H":"75.3","P":"1034","S":"11","T":"7.8","V":"40000","W":"8","Pt":"R","Dp":"3.8","$":"720"},
  {"D":"W","H":"73.3","P":"1033","S":"16","T":"8.4","V":"40000","W":"8","Pt":"F","Dp":"4.0","$":"780"},
  {"D":"W","H":"70.3","P":"1033","S":"16","T":"8.8","V":"50000","W":"3","Pt":"F","Dp":"3.8","$":"840"},
  {"D":"W","H":"68.2","P":"1034","S":"10","T":"8.6","V":"55000","W":"3","Pt":"F","Dp":"3.2","$":"900"},
  {"D":"W","H":"68.3","P":"1034","S":"8","T":"8.7","V":"50000","W":"7","Pt":"R","Dp":"3.3","$":"960"},
  {"D":"W","H":"72.7","P":"1035","S":"9","T":"7.9","V":"45000","W":"8","Pt":"R","Dp":"3.4","$":"1020"},
  {"D":"W","H":"74.8","P":"1035","S":"7","T":"7.9","V":"45000","W":"8","Pt":"R","Dp":"3.8","$":"1080"},
  {"D":"SW","H":"78.0","P":"1035","S":"6","T":"7.3","V":"30000","W":"7","Pt":"R","Dp":"3.8","$":"1140"},
  {"D":"WSW","H":"80.9","P":"1035","S":"6","T":"7.4","V":"30000","W":"8","Pt":"R","Dp":"4.4","$":"1200"},
  {"D":"WSW","H":"82.1","P":"1035","S":"6","T":"7.1","V":"30000","W":"8","Pt":"R","Dp":"4.3","$":"1260"},
  {"D":"SW","H":"84.4","P":"1035","S":"7","T":"6.9","V":"29000","W":"8","Pt":"R","Dp":"4.5","$":"1320"},
  {"D":"WSW","H":"88.0","P":"1035","S":"7","T":"6.5","V":"24000","W":"8","Pt":"R","Dp":"4.7","$":"1380"}]},
{"type":"Day","value":"2020-01-05Z","Rep":[
  {"D":"SW","H":"89.3","P":"1035","S":"7","T":"6.6","V":"23000","W":"8","Pt":"F","Dp":"5.0","$":"0"},
  {"D":"SW","H":"90.6","P":"1035","S":"7","T":"6.9","V":"26000","W":"8","Pt":"F","Dp":"5.5","$":"60"},
  {"D":"WSW","H":"91.3","P":"1034","S":"6","T":"6.9","V":"29000","W":"8","Pt":"F","Dp":"5.6","$":"120"},
  {"D":"SW","H":"90.0","P":"1035","S":"8","T":"7.1","V":"40000","W":"8","Pt":"F","Dp":"5.6","$":"180"},
  {"D":"SW","H":"90.6","P":"1035","S":"8","T":"7.0","V":"40000","W":"8","Pt":"R","Dp":"5.6","$":"240"},
  {"D":"SW","H":"94.5","P":"1034","S":"10","T":"6.6","V":"2000","W":"11","Pt":"F","Dp":"5.8","$":"300"},
  {"D":"SW","H":"93.9","P":"1034","S":"11","T":"6.5","V":"30000","W":"7","Pt":"F","Dp":"5.6","$":"360"},
  {"D":"SSW","H":"90.7","P":"1034","S":"9","T":"6.8","V":"45000","W":"8","Pt":"F","Dp":"5.4","$":"420"},
  {"D":"SSW","H":"90.6","P":"1034","S":"9","T":"6.6","V":"45000","W":"8","Pt":"F","Dp":"5.2","$":"480"},
  {"D":"SW","H":"87.4","P":"1034","S":"8","T":"6.9","V":"35000","W":"7","Pt":"F","Dp":"5.0","$":"540"},
  {"D":"SW","H":"83.9","P":"1034","S":"9","T":"7.7","V":"45000","W":"8","Pt":"F","Dp":"5.2","$":"600"},
  {"D":"SW","H":"78.2","P":"1034","S":"11","T":"8.4","V":"50000","W":"8","Pt":"R","Dp":"4.9","$":"660"},
  {"D":"WSW","H":"72.3","P":"1033","S":"15","T":"8.9","V":"60000","W":"7","Pt":"F","Dp":"4.3","$":"720"}]}]}}}}
```

We need to design a data model that will capture two relations:

- Weather location
- Weather observation

##### Weather Location

##### Weather Observation
<table>
    <thead>
        <tr>
            <th>Parameter Name</th>
            <th>Units</th>
            <th>Python data type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Location ID</td>
            <td>-</td>
            <td>int (FK)</td>
        </tr>
        <tr>
            <td>Timestamp</td>
            <td>-</td>
            <td>datetime.datetime</td>
        </tr>
        <tr>
            <td>Wind Gust (G)</td>
            <td>mph</td>
            <td>float</td>
        </tr>
        <tr>
            <td>Temperature (T)</td>
            <td>C</td>
            <td>float</td>
        </tr>
        <tr>
            <td>Visibility (V)</td>
            <td>m</td>
            <td>int or string</td>
        </tr>
        <tr>
            <td>Wind Direction (D)</td>
            <td>compass</td>
            <td>string</td>
        </tr>
        <tr>
            <td>Wind Speed (S)</td>
            <td>mph</td>
            <td>float</td>
        </tr>
        <tr>
            <td>Weather Type (W)</td>
            <td>-</td>
            <td>int (See Datapoint Table)</td>
        </tr>
        <tr>
            <td>Pressure (P)</td>
            <td>hpa</td>
            <td>float</td>
        </tr>
        <tr>
            <td>Dew Point (Dp)</td>
            <td>C</td>
            <td>float</td>
        </tr>
        <tr>
            <td>Screen Relative Humidity (H)</td>
            <td>%</td>
            <td>float</td>
        </tr>
    </tbody>
</table>