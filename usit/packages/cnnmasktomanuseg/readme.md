CNN Masks to Manuseg Segmentation
---------------------------------


### Usage

cnnmasktomanuseg.py input.ext output_directory

The input file is read and processed based on the circular boundary finding algorithm descsribed in the paper below.
The input is striped from it's extension and parameter files for inner (pupillary), `output_directory/input.inner.txt`, and outer (sclera), `output_directory/input.outer.txt`,  boundary of the iris are written into the output directory in a format compatible with `manuseg`.


### License

The USIT License applies.

If this software is used to prepare for an article please include the following reference:

#### Text

Heinz Hofbauer, Ehsaneddin Jalilian, and Andreas Uhl. “Exploiting superior CNN-based iris segmentation for better recognition accuracy”, Pattern Recognition Letters 120, 2019. DOI: 10.1049/iet-bmt.2015.0069 .

#### Bibtex

    @article{ Hofbauer19a,
        doi = {10.1016/j.patrec.2018.12.021},
        author = {Heinz Hofbauer, Ehsaneddin Jalilian, Andreas Uhl},
        title = {Exploiting superior CNN-based iris segmentation for better recognition accuracy},
        journal = {Pattern Recognition Letters},
        issn = {0167-8655},
        volume = {120},
        year = {2019},
        pages = {17-23},
    }

