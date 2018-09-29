Nuevo custom dataset Messidor-2:

Continuando con el análisis de los datasets messidor-1 y messidor-2, podríamos decir que messidor-1 es un subset de messidor-2;
la intersección entre ambos tiene 1058 imágenes, o sea que sólo 129 imágenes de messidor-1 no están en messidor-2.
En 70 (setenta) imágenes de éstas 1058 se presentan diferencias que afectan la clasificación binaria, es decir en messidor-1
la imágen se classifica como DR nivel <=1 y en messidor-2 como DR nivel >=2, o viceversa.

Armo un nuevo dataset de Messidor-2 reemplazando las 1058 imágenes .png de Messidor-2 original con las mismas imágenes
.tif de Messidor-1, y los archivos de este nuevo dataset son IMAGESv2.part0[1-4].rar. En estos archivos .rar agregué las 1200
imágenes .tif de Messidor-1 aunque sólo para 1058 tenga los labels, simplemente para simplificar el armado de los .rar.
Las annotations del nuevo dataset Messidor-2 sigue siendo las originales y están en el archivo messidor_data.csv.