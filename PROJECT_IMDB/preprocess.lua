local Source = io.input("movies.json"):read("*a")
local Json = require("dkjson")

Source = Json.decode(Source)

Genres = {}

local function genreExists(genreName)
	for i = 1, #Genres do
		if Genres[i] == genreName then
			return true
		end
	end
	return false
end

Output = {}

for i = 1, #Source do
	-- Check whether the film has necessary data
	if not(Source[i]['Title']) then
		print(string.format("Film %d [] Removed because: No Title", i))
	elseif not(Source[i]['IMDB Rating']) then
		print(string.format("Film %d [%s] Removed because: No IMDB Rating", i, Source[i]["Title"]))
    elseif not(Source[i]['IMDB Votes']) then
		print(string.format("Film %d [%s] Removed because: No IMDB Votes", i, Source[i]["Title"]))
	elseif not(Source[i]['Worldwide Gross']) then
		print(string.format("Film %d [%s] Removed because: No Worldwide Gross", i, Source[i]["Title"]))
	elseif not(Source[i]['Running Time min']) then
		print(string.format("Film %d [%s] Removed because: No Running Time min", i, Source[i]["Title"]))
	elseif not(Source[i]['Release Date']) then
		print(string.format("Film %d [%s] Removed because: No IMDB Votes", i, Source[i]["Title"]))
	elseif not(Source[i]['Production Budget']) then
		print(string.format("Film %d [%s] Removed because: No IMDB Votes", i, Source[i]["Title"]))
	elseif not(Source[i]['Major Genre']) then
		print(string.format("Film %d [%s] Removed because: No Major Genre", i, Source[i]["Title"]))
	else
		if not(genreExists(Source[i]['Major Genre'])) then
			Genres[#Genres + 1] = Source[i]['Major Genre']
			print("New Genre Found: "..Source[i]['Major Genre'])
		end
		Output[#Output + 1] = Source[i]
	end
end

print(string.format("Finished: %d Valid films with %d genres", #Output, #Genres))
io.output("movies_processed.json"):write(Json.encode(Output))
print(Json.encode(Genres))
