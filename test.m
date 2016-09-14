catName = {'bed', 'chair', 'sofa', 'table', 'counter', 'desk', 'lamp', 'pillow', 'sink', 'garbage-bin', 'television','monitor', 'dresser', 'night-stand', 'door', 'bathtub', 'toilet', 'box', 'bookshelf'};
catName = sort(catName);
allclassNum=length(catName);

%处理全部的GT，对一张图片进行语义分割处理
c=benchmarkPaths();
allcolors=extractcolors(allclassNum+1);%多一个颜色用作意外处理
allclassname_=dt.allClassName;
mapclass_=dt.mapClass;
name2color_map = containers.Map;
num2cls_map=containers.Map;

for i=1:allclassNum
     f_index= find(strcmp(allclassname_, catName{i}));
     if isempty(f_index)
         index(i)=-1;
     else
         index(i)=f_index;
     end
    colordatabase{i}.color=allcolors{i};
    colordatabase{i}.cls=catName{i};
    colordatabase{i}.index=index(i);
    num_key=num2str(index(i));
    num2cls_map(num_key)=catName{i};
    name2color_map(catName{i})=allcolors{i};%从类标到颜色的一个map
end
    
%构建出一张hash表
cls2num_map = containers.Map(catName,index);
databasemap.name2color_map=name2color_map;
databasemap.num2cls_map=num2cls_map;
databasemap.cls2num_map=cls2num_map;

save('/home/project/rcnn-depth-master/eccv14-data/benchmarkData/metadata/colordatabase','colordatabase','databasemap');