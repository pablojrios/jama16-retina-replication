Nuevo custom dataset Messidor-2:
Comentarios marzo 2019:

Analizando nuevamente (marzo 2019) las diferencias entre messidor-1 (aka 'messidor original') y messidor-2
(aka 'messidor extension'), concluyo que messidor-1 NO es un subset de messidor-2 y que la relación es que la
intersección entre ambos tiene 1058 imágenes que son all image pairs from the original Messidor dataset, that is
529 examinations. Esto está explicado en el ReadMe.txt~ en IMAGES.part[1..4].rar de messidor-2.
O sea que sólo 142 imágenes de messidor-1 no están en messidor-2, y estas son las imágenes que no están de a pares.
Y de estas 142 imágenes, 13 están duplicadas (ver messidor.sh).

En 70 (setenta) imágenes de éstas 1058 se presentan diferencias que afectan la clasificación binaria, es decir en messidor-1
la imágen se classifica como DR nivel <=1 y en messidor-2 como DR nivel >=2, o viceversa.

Armo un nuevo dataset de Messidor-2 reemplazando las 1058 imágenes .png de Messidor-2 original con las mismas imágenes
.tif de Messidor-1, y los archivos de este nuevo dataset son IMAGESv2.part0[1-4].rar. En estos archivos .rar agregué las 1200
imágenes .tif de Messidor-1 aunque sólo para 1058 tenga los labels, simplemente para simplificar el armado de los .rar.
Las annotations del nuevo dataset Messidor-2 sigue siendo las de Messidor-2 y están en el archivo messidor_data.csv.

Finalmente armamos el archivo messidor-2_data.csv a partir de messidor_data.csv agregandole la columna reflejos para poder
excluir todas las imágenes IM00nnnn.JPG que son las nuevas de messidor-2 y que la gran mayoría tienen reflejos.