function draw_box(t1,t2,constriction,target,omega)

hold on;
    
    if omega > 0
    
    x1 = t1;
    x2 = t2;
    
    y1 = constriction - 1;
    y2 = constriction;
    
    x = [x1 x2 x2 x1 x1];
    y = [y1 y1 y2 y2 y1];
    
    %fill(x,y,[0.8 0.8 0.8]);
    h = fill(x,y,1 - 0.5*omega/50*[1 1 1]);
    set(h,'EdgeColor','none')
    
    text((x1+x2)/2, (y1+y2)/2,sprintf('(%4.2f, %4.2f)',omega,target*2.4), ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'middle');
 
    end

end

