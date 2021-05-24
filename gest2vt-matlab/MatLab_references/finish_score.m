function finish_score(N);

ticks=0.5:1:6;
yticks(ticks);
yticklabels({'bilabial','alveolar','palatal','velar','pharyngeal','velopharyngeal'});
axis([0 N 0 6]);

plot([0 0 N N 0],[0 6 6 0 0],'k')