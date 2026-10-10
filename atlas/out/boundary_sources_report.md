# Boundary sources for colonial India census maps (1872–1951)

Research date: 2026-10-09. Every link below was tested on that date (WebFetch, curl, the Harvard Dataverse API, the GitHub API, the ArcGIS REST API, the Zenodo/figshare/Mendeley APIs, or the OpenHistoricalMap Overpass API) unless it is marked **UNVERIFIED**.

## Bottom line

1. **No public, downloadable vector dataset of district boundaries covering all of British India (including present-day Pakistan, Bangladesh and Burma) exists for any census year from 1872 to 1941.** Every academic project that built one either kept it private or shared only tables:
   - Fenske, Kala & Wei converted the 1931 census district map to a shapefile but did not release it.
   - The Appraising Risk project made 1872, 1881 and 1891 shapefiles but has not published them.
   - Donaldson's AER 2018 paper has no data package.
2. **The biggest find is a table, not a map.** India State Stories' *India State and District Evolution Database, 1872–2025* (Harvard Dataverse, CC BY 4.0) includes `district_evolution_1872_1941.xlsx`. It holds 3,220 source-to-destination district links for every census pair from 1872→1881 through 1931→1941, covering British districts, princely states and princely districts. This replaces most of our hand-compiled `district_changes.csv` (16 rows) and `unit_periods.csv` (13 rows) for present-day India. It does **not** cover Pakistan, Bangladesh or Burma.
3. **A free, CC0 1951 layer is already in the same GitHub release we use.** The release `census/historical` also contains `India-State-Districts-1951` (as well as 1961–2001), so 1951 needs no new digitizing for present-day India. The same Dataverse deposit adds `district_consolidation_1941_1951.xlsx`, which crosswalks the 1941 units to the 1951 units.
4. **For whole-empire geometry, the best available option is to georeference the Imperial Gazetteer atlases ourselves.** The 1909 and 1931 atlas sheets (including Burma, Sind, NWFP, Baluchistan and Bengal) are served by DSAL at full resolution through IIIF (provincial sheets are about 4950×4096 px). Our existing 1941 map is about 15 km off, so this would also improve on it.
5. **Usable partial vectors exist:**
   - **Punjab 1931:** an ArcGIS feature service with all 29 British Punjab districts (both sides of the 1947 border), Delhi and the Punjab States. It can be queried publicly but has no license stated.
   - **Provinces 1891–1941:** the `wtennant/plague_india` GitHub repository has province-level shapefiles for the whole empire, including Burma, split into British and native territory, for 1891, 1901, 1911, 1921, 1931 and 1941.
6. **For the Pakistan/Bangladesh side of the change tables:** a 1965 PIDE report, *A Statistical History of District Boundary Changes in Pakistan, 1881–1961*, covers East and West Pakistan (76-page scanned PDF on figshare, CC BY-NC-ND 3.0). It is the textual source to pair with the India State Stories table.

---

## 1. Candidate vector boundary datasets

### 1.1 India State Stories (FLAME University): historical state/district shapefiles 1941–2021. **Currently used for 1941.**
- **Landing page:** https://www.indiastatestory.in/datadownloads
- **Mirror we use:** https://github.com/yashveeeeeeer/india-geodata/releases/tag/census/historical
  - Assets: `India-State-Districts-{1941,1951,1961,1971,1981,1991,2001}` in `.parquet`, `.pmtiles` and `.geojsonl.7z`.
  - Also four CSVs covering 1951–2024: district time series, splits, new districts and name changes.
- **Original Google Drive:** https://drive.google.com/drive/folders/1ZRDWsMRIC0x0YlaFrNIwN4MJafZ68dKV
  - Subfolders are `India_State-Dist-1941` and `India-State-Dist-1951` through `-2011`.
  - **There is no folder for any year before 1941.** I checked the folder listing directly.
  - The page says downloads need access requested with a verified email.
- **Years:** 1941, 1951, …, 2021.
- **Coverage:** present-day India only. The page says "1941–1971 maps are clipped to modern Indian boundaries."
- **Format:** shapefile (Drive) or parquet/pmtiles/geojsonl (mirror).
- **License:** CC0, per the mirror's `data/census/historical/README.md` and `metadata.json`. The ISS site footer says "All rights reserved", but its Dataverse deposits are CC0 or CC BY 4.0.
- **How it was made:** digitized from the Census of India *Administrative Atlas of India* (2011) maps. Their own caveat: "digitization errors may persist… for representation only."
- **Downloadable:** yes.
- **Action item:** the 1951 layer is in the same release and is not used yet. The atlas code references only `India-State-Districts-1941`.

### 1.2 Fenske, Kala & Wei, "Railways and cities in India" (JDE 161, 2023, 103038): 1931 census district map
- **Paper:** https://doi.org/10.1016/j.jdeveco.2022.103038
- **Working paper:** https://warwick.ac.uk/fac/soc/economics/research/workingpapers/2021/twerp_1349_-_fenske.pdf
- **What the paper says (section 3.3):** "we begin by converting the map of districts from the 1931 census to a shapefile. Because this map has a low resolution, we are concerned that this will lead to measurement error…"
  - The city data cover "modern-day Bangladesh, Burma, India, and Pakistan."
  - Districts are held fixed at 1931 boundaries, with some units aggregated, such as Rest of Bombay Presidency, Kathiawar agencies and Cochin.
- **The shapefile is not public.**
  - Warwick CAGE releases only tables: https://warwick.ac.uk/fac/soc/economics/research/centres/cage/data/datasets/03-02-22-indian_district_populations_by_sex_1881_1931 (direct file: https://warwick.ac.uk/fac/soc/economics/research/centres/cage/manage/datasets/ds5.2022.xlsx).
  - I downloaded it. It has 557 rows of district populations by sex for 1881–1931 on 1931 boundaries, across 25 provinces/agencies including **Burma, NWFP, Baluchistan and Bengal**. Terms require citation.
  - There is also a city dataset: https://warwick.ac.uk/fac/soc/economics/research/centres/cage/data/datasets/03-02-22-indian_city_populations_by_sex_with_latitude_and_longitude_1881_1931
  - I found no Mendeley Data deposit by API search. The authors' websites have no data page. jamesfenske.com now redirects to sites.google.com/view/james-fenske, which has no data links.
- **Recommendation:** email Fenske and ask for the 1931 district shapefile. It is the closest thing to a whole-empire layer, though the authors themselves call it low resolution. The CAGE table is a useful whole-empire **1931 district list and population check**, including Burma and Pakistan.

### 1.3 Appraising Risk / Indian Ocean World Centre (McGill): "Digitization of Indian Census Districts 1872 to Present"
- **URL:** https://www.appraisingrisk.com/2020/10/23/digitization-of-indian-census-districts-1872-to-present/ (posted 2020, last updated 2023-04-03)
- **Claims:** shapefiles for 1872, 1881 and 1891, "still being finished." Each polygon has a name and a type (British Territory or Princely State).
- **Method:** scanned census maps were enhanced with ImageMagick and georeferenced in ArcGIS with 15–30 control points, then borders were auto-traced and corrected by hand.
- **Coverage:** India. Pakistan, Bangladesh and Burma are not mentioned.
- **License:** not stated.
- **Downloadable: NO.** I crawled `/collected-data/`, `/gis-data-analysis/`, `/historical-database/`, `/maps-visualizations/`, `/south_asia/` and `/external-datasets/`. The only download is a disease spreadsheet. I found no Borealis or Dataverse deposit.
- **Recommendation:** contact IOWC (McGill) if 1872–1891 matter, but do not plan around getting it.

### 1.4 Donaldson, "Railroads of the Raj" (AER 2018), and Burgess & Donaldson (famine)
- **AEA materials:** https://www.aeaweb.org/articles?id=10.1257/aer.20101199
  - "Materials" 8686 is a PDF appendix.
  - "Materials" 8687 (https://www.aeaweb.org/content/file?id=6710) is a 65 KB zip that holds **only the disclosure statement**.
  - **There is no data or shapefile.**
- Note: "Donaldson & Hornbeck" (QJE 2016) is about US counties, not India. The colonial-India district panel (235 districts, 1870–1930) is Donaldson 2018, and its boundaries were never released.

### 1.5 Iyer (REStat 2010), Banerjee & Iyer (AER 2005), Iyer & Weir (JDE 2025)
- These papers use **post-1947 (1961-era) district units** coded as formerly British or princely. They do not use colonial polygons.
- Iyer's site (https://sites.nd.edu/lakshmi-iyer/) has no data page; `/data/` returns 404.
- I could not retrieve a REStat Dataverse deposit for Iyer 2010.
- openICPSR pages (for example project 116059) sit behind a Cloudflare challenge, so I could not list their files. **UNVERIFIED** whether any of them contain geometry. Given how these papers are designed, it is unlikely.
- Useful indirectly: India State Stories cites Banerjee & Iyer Appendix Table 1 for acquisition dates (see 2.2).

### 1.6 Gasparyan, "Indirect Rule and Public Goods Provision: Evidence from Colonial India" (Harvard Dataverse, CC0)
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HYNOCY
- Contains only border points for Hyderabad and Mysore within Karnataka (`hyd_borderpts.dbf`, `mysborderpts.dbf`). This is regional, not a district layer, so it is not useful to us.

### 1.7 Punjab 1931 districts (Penn student project, Plum Xu)
- **Feature service:** https://services3.arcgis.com/9nfxWATFamVUTTGb/arcgis/rest/services/Interactive_Punjab_Agriculture_Map_WFL1/FeatureServer (layers 8 and 9 hold the district polygons)
- **ArcGIS item:** 9e116c5090e845afb29af71c1f5916a3
- **StoryMap:** ad9df44084f7423ba7113f2aa85ae270
- **Years:** 1931.
- **Coverage:** British Punjab. That means 29 districts (Rawalpindi, Lahore, Multan, Lyallpur, Hissar, Ambala and so on, on **both sides of the 1947 border**), plus Delhi, Bahawalpur State and other princely states. There are 38 features in total.
- **Format:** ArcGIS feature service. It can be exported with `query?where=1=1&outFields=*&f=geojson`. I confirmed the record count of 38 and the attributes.
- **License:** none stated, though access is "public."
- **How it was made:** hand-traced from the Imperial Gazetteer Atlas 1931, Punjab sheet (p. 34, DSAL scan), with 1931 census data joined.
- **Downloadable:** yes, through the REST API.
- **Use:** validation, or a stopgap for Pakistan Punjab in 1921–1941. It is hand-traced and accuracy is unassessed. Ask the author (Penn) for permission before redistributing.

### 1.8 `wtennant/plague_india`: province-level shapefiles for all of British India
- **Repo:** https://github.com/wtennant/plague_india
- **Files:** `Data/Shapefiles/india_{1891,1901,1903,1904,1911,1921,1923,1931,1935,1936,1941}.shp`
- **Raw example:** https://raw.githubusercontent.com/wtennant/plague_india/master/Data/Shapefiles/india_1931.shp
- **Years:** 1891–1941, every time a province changed, including 1903, 1904, 1923, 1935 and 1936.
- **Coverage:** **the whole empire, including Burma, Sind, NWFP, Baluchistan, Bengal and Assam.** Bounds are 60.9–101.2°E and 8.1–44.6°N. There are 32 features in 1891 and 35 in 1931.
- **Attributes:** `STATE_NAME` plus a `NATIVE` flag (0 = British, 1 = native states). **Province level only, with no districts.**
- **Format:** ESRI shapefile in World Cylindrical Equal Area (ESRI:54034). Some rings have invalid winding order, which GDAL auto-fixes.
- **License:** LGPL-3.0 (the repository license).
- **How it was made:** I could not confirm the source. It accompanies Tennant et al., the plague/climate study of British India 1898–1949.
- **Downloadable:** yes. I downloaded and read the 1891 and 1931 files.
- **Use:** a province/agency outline and British-vs-native mask for every census year from 1891, including the areas missing from our 1941 map. It can serve as a "frame" so the map is not blank over Pakistan, Bangladesh and Burma.

### 1.9 OpenHistoricalMap
- **Overpass query** (I ran it against https://overpass-api.openhistoricalmap.org/api/interpreter for bounding box 6–38°N, 60–100°E):
  - About 80 colonial-era relations at admin_level 4 (provinces and large princely states). Examples: Punjab Province 1858–1901, 1901–1911 and 1911–1947; NWFP 1901; Sind 1936; Bengal Presidency 1912; Bihar and Orissa 1912–1936; Hyderabad; Jammu & Kashmir; Gwalior; Kalat; and a few dozen individual princely states.
  - Only **about 13 district-level relations**, such as Ganjam, Vizagapatam, Nilgiris, North/South Canara, Malabar and Kutch.
- **License:** CC0.
- **Coverage:** sparse. The contributor forum (https://community.openstreetmap.org/t/indias-history-on-openhistoricalmap/124585) confirms there is "very little coverage."
- **Use:** not a district source. It might cross-check province edges.

### 1.10 ML Infomap "India HistoricMap" (1951–2001)
- **Harvard (Harvard affiliates only):** https://geo.nyu.edu/catalog/harvard-indiahist-district51
- **Stanford (Stanford affiliates only):** https://purl.stanford.edu/mb214hn2157 (14 layers, 1951–2011); 1951 layer at https://purl.stanford.edu/tf374bd2484
- **Format:** shapefile at 1:1M; the 1951 layer has 467 polygons.
- **License:** commercial, restricted, educational use only.
- **Coverage:** India only. **No colonial years.**
- Penn's ML Infomap guide (https://guides.library.upenn.edu/IndiaGISdatasets) starts at 1981.
- **Use:** restricted. The ISS 1951 layer (1.1) is the open alternative.

### 1.11 Vanneman & Barnes Indian District Database
- **URL:** http://vanneman.umd.edu/districts/index.html (expired TLS certificate; reachable with `curl -k`)
- **Coverage:** **1961–1991** tables with boundary-adjusted units. No colonial years, and no geometry on the site.
- **Not relevant** to 1872–1951.

### 1.12 Other items checked (not useful, or not verifiable)
- **Kumar & Somanathan district crosswalks:** these cover 1971–2001 (and 1961) only, per the working paper abstract surfaced in search. Not colonial.
- **SHRUG / Development Data Lab, IPUMS-International, DataMeet, GADM, geoBoundaries:** modern or post-1951 only. IPUMS says it built Indian boundaries from GADM, statoids and the 2011 Administrative Atlas (https://forum.ipums.org/t/source-shapefile-india/4997). None is a colonial source.
- **Harvard Dataverse API searches** (for "British India districts", "colonial India shapefile", "princely states shapefile", "India 1931 census districts" and similar): only ISS deposits and irrelevant hits. One example is "HGC Districts SHP," which turned out to be Crimean points.
- **Zenodo and figshare API searches:** no colonial India boundary datasets.
- **H-Net thread "British India ArcGIS shapefile"** (https://networks.h-net.org/node/22055/discussions/76752/british-india-arcgis-shapefile): **UNVERIFIED** (returns 403). Per search snippets, a commenter made a rough province-level shapefile from the 1909 Gazetteer and "would not recommend" it. Also mentions a 1922 Rumsey political map.
- **AlternateHistory.com "Historical GIS resources" thread:** has "1948 British India administrative divisions" shapefiles, a hobbyist source. **UNVERIFIED** (returns 403). Not citable.
- **GaDOE "Princely States" ArcGIS layer** (item 48ffd87c674c4402a564d12eff9115e3): 278 **points**, an educational layer. Not polygons.
- **Gupta-era / Schwartzberg *Historical Atlas of South Asia* (DSAL):** raster plates only. A 2006 promise of a GIS interface was apparently never delivered.

### 1.13 Raster sources to digitize from (not vectors)
- **DSAL Imperial Gazetteer Atlas 1931, IIIF:**
  - Manifest: https://dsal.uchicago.edu/reference/gaz_atlas_1931/manifests/gazetteer_atlas_1931.json
  - 73 canvases. Provincial sheets include Bengal; Bihar & Orissa; Assam; United Provinces; Punjab; NWFP & Kashmir; Rajputana; Baluchistan; Bombay/Sind (two sections); Central India; Central Provinces; Hyderabad; Madras (two sections); and **Burma (three sections)**. There is also "Political Divisions of the Indian Empire."
  - Full resolution is at `https://dsal.uchicago.edu/iiif/gazetteer/1931/<file>.tif/full/max/0/default.jpg` (for example, Punjab is 4950×4096). I confirmed it returns HTTP 200.
  - Attribution: DSAL. The underlying maps are Bartholomew/Government of India publications from 1909 and 1931.
- **DSAL Imperial Gazetteer Atlas 1909, IIIF:** https://dsal.uchicago.edu/reference/gaz_atlas_1909/manifests/gazetteer_atlas_1909.json (HTTP 200). Its boundaries are about right for 1901 and 1911.
- **Census of India 2011 *Administrative Atlas of India* (national volume):** 15 maps, one per census from 1872 to 2011, all drawn within present-day India's boundary. This is the source ISS digitized. Fenske once hosted `final_atlas_india_2011.pdf` at Warwick, but that link now returns 404. The census NADA catalog has per-state volumes.
- **Burma:** Zenodo 13346102 (https://zenodo.org/records/13346102), Horst Held, CC BY 4.0. About 555 georeferenced Survey of India "half-inch" (1:126,720) sheets of Burma from 1878–1949, as GeoTIFF/JPG plus KML and an Excel master list (8.1 GB). Survey sheets show district boundaries, so this is the best georeferencing control for Burma.
- **National Library of Scotland Survey of India series** (https://maps.nls.uk/india/survey-of-india/): **UNVERIFIED**, because the site served a CAPTCHA. The OSM forum (July 2026) reports that NLS released 1-inch India sheets under CC BY.
- **Wikimaps Warper:** a georeferenced 1909 Punjab sheet (search snippet only). **UNVERIFIED.**

---

## 2. Machine-readable tables of district creation, abolition and renaming (1872–1947)

### 2.1 India State and District Evolution Database (ISDED), 1872–2025. **RECOMMENDED**
- **Landing page:** https://doi.org/10.7910/DVN/D1AGUR (Harvard Dataverse, Jolad, Kalra & Singh). **CC BY 4.0.**
- **Direct files** (Dataverse access API):
  - `district_evolution_1872_1941.xlsx`: https://dataverse.harvard.edu/api/access/datafile/13322421
  - `district_consolidation_1941_1951.xlsx`: https://dataverse.harvard.edu/api/access/datafile/13322425
  - `district_proliferation_1951_2024.xlsx`: https://dataverse.harvard.edu/api/access/datafile/13322422
  - `admin_units_2025.xlsx`, plus four `codebook.pdf` files.
- **Structure of `district_evolution_1872_1941`:**
  - 3,220 rows. Columns: `source_admin_div`, `source_type`, `source_district`, `dest_admin_div`, `dest_type`, `dest_district`, `source_year`, `dest_year`, `filter_state`.
  - There are 460 links for each census pair: 1872→81, 81→91, 91→1901, 1901→11, 11→21, 21→31 and 31→41.
  - Unit counts by year: 340 (1872), 404 (1881), 401 (1891), 401 (1901), 403 (1911), 412 (1921), 412 (1931), 414 (1941).
  - Splits and merges by step (one source to many destinations, or the reverse):
    | Step | Splits | Merges |
    |---|---|---|
    | 1872→81 | 34 | 7 |
    | 1881→91 | 2 | 5 |
    | 1891→1901 | 1 | 1 |
    | 1901→11 | 5 | 3 |
    | 1911→21 | 6 | 2 |
    | 1921→31 | 1 | 1 |
    | 1931→41 | 5 | 4 |
  - The table keeps period spellings, so it doubles as a historical-name crosswalk. Examples: Seebsaugor→Sibsagar, Bhaugulpore→Bhagalpur, Tirhoot→Darbhanga+Muzaffarpur, Hooghly with Howrah→Hooghly+Howrah.
- **Types covered:** British districts, princely states, princely districts (for example, Baroda's prants), NEFA and tribal areas.
- **Coverage:** **present-day India only.** "Punjab" stops at the Indian side, and there are no rows for Lahore, Karachi, Dacca, Chittagong, Rangoon or Peshawar. 18 `filter_state` groups.
- **How it was made:** digitized from the Administrative Atlas of India (1872–2011), supplemented by census reports, Survey of India maps and Imperial and provincial gazetteers.
- **Limitation:** links are whole-unit parent→child. They do not record transfers of territory between districts that both continue to exist, and they give no exact dates within a decade (the year is the census year).
- **Downloadable:** yes. I downloaded and parsed it.

### 2.2 "Making of British Provinces and Districts" (ISS, Harvard Dataverse, CC0)
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZVPZAN
- **File:** `Formation of British provinces and districs- excluding Princely states.xlsx` (https://dataverse.harvard.edu/api/access/datafile/13956376), plus a codebook PDF.
- **Contents:** about 40 rows giving province, region, date and mode of British acquisition, the colonial districts involved, and their modern equivalents, for **1690–1862**.
- **Sources:** Baden-Powell 1892; Administrative Atlas; Banerjee & Iyer 2005, Appendix Table 1.
- **Use:** pre-1872 context, such as the dates territory came under direct rule. It does not cover 1872–1947 changes.

### 2.3 "Database of Princely States in British India 1939" (ISS, Harvard Dataverse, CC0)
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2ES4C1
- **File:** https://dataverse.harvard.edu/api/access/datafile/13958718 (`.xlsx`, 594 rows)
- **Fields:** agency and province, sub-group, state or estate name, area (sq mi), 1931 census population, revenue, gun salutes, ruler and religion.
- **Source:** *Memoranda on the Indian States, 1939.*
- **Coverage:** all princely states, **including those now in Pakistan**, such as Bahawalpur, Khairpur and Kalat. Not geometry.
- **Use:** a master list and population check for princely states in the 1931 and 1941 maps.

### 2.4 PIDE 1965: *A Statistical History of District Boundary Changes in Pakistan, 1881–1961* (Farooq, Hussain & Sanaullah)
- **Landing page:** https://opendocs.ids.ac.uk/articles/report/A_statistical_history_of_district_boundary_changes_in_Pakistan_1881-1961/26480101 (returns 403 to WebFetch, but the figshare API works)
- **Direct PDF:** https://ndownloader.figshare.com/files/48229993 (13.3 MB, 76 pages, scanned, with a rough OCR text layer)
- **License:** CC BY-NC-ND 3.0.
- **Contents:** **East and West Pakistan.** Tables of net area change by district from 1881 to 1961, and changes broken down by cause, organized by census decade.
- **Use:** the only systematic source found for the change table of present-day Pakistan and Bangladesh. It would need transcription, which our OCR pipeline could do.

### 2.5 Also checked
- ISS 1951–2024 CSVs in the india-geodata repository (`data/census/historical/*.csv`): post-1951 only.
- Wikipedia "Districts of British India": a narrative list, not structured, so not citable as data.
- CAGE district populations 1881–1931 (1.2): implicitly fixes the 1931 district list for the **whole empire**, which is useful for Burma, Sind and NWFP name lists.
- No machine-readable change table for Burma was found.

---

## 3. Ranked recommendation by census year

General plan:
1. **Present-day India:** keep the ISS 1941 polygons as the base geometry and **back-cast** earlier years by dissolving 1941 polygons along the ISDED parent→child links (2.1). This replaces the hand-compiled change tables. Report the result as "1941 geometry aggregated to period units", not as true period boundaries.
2. **Pakistan, Bangladesh and Burma:** no district vectors exist. Fill these areas with GADM 4.1 units dissolved to colonial districts, using the PIDE 1965 report (Pakistan/Bangladesh) and the CAGE 1931 district list. Outline them with the `plague_india` province polygons.
3. **Upgrade path for the paper:** georeference the DSAL 1909 and 1931 atlas provincial sheets through IIIF (with the Zenodo Burma sheets as control) and trace district lines. This gives a whole-empire layer for 1901/1911 (from 1909) and 1921/1931/1941 (from 1931). It also fixes the roughly 15 km error in the 1941 layer, and the ISDED links let it cascade to the other years.

| Census year | Present-day India | Pakistan / Bangladesh | Burma | Province/agency frame |
|---|---|---|---|---|
| **1872** | (1) ISS 1941 dissolved via ISDED 1872→81 links, which have the most changes (34 splits), so manual QA is needed. (2) Ask Appraising Risk for their 1872 file. | GADM dissolved using the PIDE report (from 1881; 1872 needs census report notes). | Lower Burma only, part of British India. GADM dissolve; low priority. | `plague_india` begins in 1891. Use the 1891 frame with known 1872–91 edits (for example, Assam separated from Bengal in 1874). |
| **1881** | (1) ISDED back-cast. (2) Appraising Risk (unreleased). | GADM + PIDE (1881 baseline). | GADM dissolve (Lower Burma). | Same as 1872. |
| **1891** | (1) ISDED back-cast. (2) Appraising Risk (unreleased). | GADM + PIDE. | GADM dissolve; Upper Burma included from 1891. | `plague_india/india_1891.shp` |
| **1901** | (1) ISDED back-cast. (2) **Best upgrade:** trace the DSAL 1909 atlas. | GADM + PIDE (note NWFP created 1901). (Upgrade: 1909 atlas.) | GADM dissolve (upgrade: 1909 atlas + Zenodo sheets). | `india_1901.shp` |
| **1911** | (1) ISDED back-cast. (2) DSAL 1909 atlas trace, with 1911 changes from ISDED (Bengal reunified, Bihar & Orissa, Delhi). | GADM + PIDE. (Upgrade: 1909 atlas plus edits.) | Same as 1901. | `india_1911.shp` |
| **1921** | (1) ISDED back-cast. (2) DSAL 1931 atlas trace. | GADM + PIDE. **Punjab:** Penn 1931 layer (1.7) as reference. | GADM / 1931 atlas. | `india_1921.shp` |
| **1931** | (1) ISDED back-cast, i.e. near-identity with 1941 (19 changed links). (2) **DSAL 1931 atlas trace**, the most valuable single digitization. (3) Fenske–Kala–Wei 1931 shapefile, if the authors share it. | (1) Fenske–Kala–Wei shapefile, if shared. (2) Penn Punjab 1931. (3) GADM + PIDE. | (1) Fenske–Kala–Wei, if shared. (2) DSAL 1931 Burma sheets. | `india_1931.shp` |
| **1941** | (1) **ISS 1941 (current), CC0.** (2) Replace with the 1931-atlas trace plus ISDED 1931→41 edits if positional accuracy matters. | GADM + PIDE. Penn Punjab 1931 for Punjab (unchanged districts). | GADM / 1931 atlas (Burma separated in 1937; include only if the 1941 tables cover it). | `india_1941.shp` |
| **1951** | (1) **ISS 1951 layer, CC0, already in the `census/historical` release** (`India-State-Districts-1951.parquet`). Use it with `district_consolidation_1941_1951.xlsx`. (2) ML Infomap 1951 (restricted; avoid). | West Pakistan and East Bengal 1951: GADM dissolved using the PIDE report (it runs to 1961). | Not part of the census. | n/a |

### Licensing summary (for the paper's data statement)

| Source | License | Can we redistribute derived polygons? |
|---|---|---|
| ISS 1941/1951 polygons (via india-geodata) | CC0 | Yes |
| ISDED tables | CC BY 4.0 | Yes, with citation (Jolad, Kalra & Singh 2026, doi:10.7910/DVN/D1AGUR) |
| ISS princely-states and provinces tables | CC0 | Yes |
| `plague_india` province shapefiles | LGPL-3.0 | Yes, with the license notice. Source provenance is unclear, so cite as secondary. |
| Penn Punjab 1931 | Unlicensed | Reference and validation only, unless the author grants permission |
| PIDE 1965 report | CC BY-NC-ND 3.0 | Cite the facts; do not redistribute the PDF |
| DSAL atlas scans | Attribution "DSAL, University of Chicago". Underlying 1909/1931 maps are old enough that copyright has almost certainly expired (Crown copyright lasts 50 years from publication) | Yes for our own traced vectors, crediting DSAL |
| Zenodo Burma sheets | CC BY 4.0 | Yes, with attribution |
| GADM 4.1 | Free for academic use; no commercial redistribution | Already in use |

### Things I could not verify
- Whether any openICPSR replication package (Cloudflare-blocked) contains colonial geometry. This applies to Iyer 2010, Chaudhary & Fenske's "Railways, development and literacy" (openICPSR, 2023) and others. They are probably tabular only.
- The contents of the H-Net and AlternateHistory threads (both returned 403).
- The NLS Survey of India viewer and its license (CAPTCHA).
- The Wikimaps Warper georeferenced 1909 Punjab sheet.
- The original source of the `plague_india` province polygons.
- Whether the Fenske–Kala–Wei 1931 shapefile or the Appraising Risk 1872–1891 shapefiles would be shared on request.
