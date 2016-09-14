load('output.mat');
im=imread('images.png');
%imshow(im);
%a=O(2);
%a=cell2mat(a);
%imshow(ucms);
%imshow(ucm2);
% sc=candidates.scores;%
% [sc2,I]=sort(sc,'descend');
% a=candidates.bboxes;
im=imresize(im,2);
sp = bwlabel(ucm2 < 0.20);
sp=sp(2:1:end, 2:1:end);
%extract the superpixel area
SuperPixelNum=max(max(sp));
imagesc(im);

%fill the sp with zeros
% for sprow=2:size(sp,1)
%     for spcol=2:size(sp,2)
%         if sp(sprow,spcol)==0
%             if sp(sprow-1,spcol)~=0
%                 sp(sprow,spcol)=sp(sprow-1,spcol);
%             end
%             if sp(sprow+1,spcol)~=0
%                 sp(sprow,spcol)=sp(sprow+1,spcol);
%             end
%             if sp(sprow,spcol-1)~=0
%                 sp(sprow,spcol)=sp(sprow,spcol-1);
%             end
%             if sp(sprow,spcol+1)~=0
%                 sp(sprow,spcol)=sp(sprow,spcol+1);
%             end
%         end
%     end
% end


% for i = 1:3, csp(:,i) = accumarray(sp(:), linIt(im(:,:,i)), [], @mean); end 
%newim=im;
%imagesc(ind2rgb(sp, im2double(uint8(im))));
abc=ind2rgb(sp, im2double(uint8(im)));
for i=1:SuperPixelNum
    [row,col]=find(sp==i);
    
    newim=zeros(size(im,1),size(im,2),size(im,3));
    %show the image
    for rr_=1:length(row)
        a=im(row(rr_),col(rr_),:);
        newim(row(rr_),col(rr_),:)=im(row(rr_),col(rr_),:);
    end
    imshow(uint8(newim));

    boxTop=min(row);
    boxBottom=max(row);
    boxLeft=min(col);          
    boxRight=max(col);
    %imagesc(abc); 
    %rectangle('Position',[boxLeft,boxTop,boxRight-boxLeft,boxBottom-boxTop]);
    
    %cut the image to the size of the sp
    newim=newim(boxTop:boxBottom,boxLeft:boxRight);
    % imshow(uint8(newim));
    pause;close all;
end
%mysuperpixels=candidates.superpixels;