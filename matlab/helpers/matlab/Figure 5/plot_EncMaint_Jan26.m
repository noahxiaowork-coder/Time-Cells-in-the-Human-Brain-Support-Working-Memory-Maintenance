function plot_EncMaint(nwbAll, all_units, neural_data_file)
% -------------------------------------------------------------------------
% Visualises ALL encoding phases (E1‑L1, E1‑L2, E1‑L3, E2‑L2, E2‑L3, E3‑L3)
% plus Maintenance‑L1, for datasets **without preferred_image**.
%
% Each neuron contributes 7 rows; one min–max range is applied across its
% whole concatenated vector.  Rows are sorted by the latency of the
% Maintenance‑L1 peak (late → early) so they align across panels.
%
% Xiaoxuan (Noah) Xiao — Apr 2025
% -------------------------------------------------------------------------

    % ---------- user‑tunable parameters ----------------------------------
    binSzEnc   = 0.1;         % 100 ms bins for encoding PSTHs
    encWindow  = 1.0;         % 1 s window length
    numBinsEnc = round(encWindow/binSzEnc);
    numBinsMnt = 25;          % Maintenance vector length
    tsKeys     = {'timestamps_Encoding1', ...
                  'timestamps_Encoding2', ...
                  'timestamps_Encoding3'};

    % ---------- load pre‑processed structure -----------------------------
    load(neural_data_file,'neural_data');
    N = numel(neural_data);

    % pre‑allocate (#neurons × #bins) matrices
    E1L1 = nan(N,numBinsEnc); E1L2 = E1L1;   E1L3 = E1L1;
    E2L2 = E1L1;              E2L3 = E1L1;   E3L3 = E1L1;
    M1   = nan(N,numBinsMnt);

    % ---------- main loop ------------------------------------------------
    for n = 1:N
        nd = neural_data(n);
        tri = nd.trial_imageIDs;                 % (#trials × 3)

        % locate SU metadata
        suIdx = find([all_units.subject_id]==nd.patient_id & ...
                     [all_units.unit_id]==nd.unit_id,1);
        if isempty(suIdx), continue, end
        SU = all_units(suIdx);

        % onset timestamps for Enc1‑3
        ts = cell(1,3);
        for e = 1:3
            ref = tsKeys{e};
            if isKey(nwbAll{SU.session_count}.intervals_trials.vectordata, ref)
                ts{e} = nwbAll{SU.session_count}.intervals_trials. ...
                        vectordata.get(ref).data.load();
            end
        end

        % load‑specific trial indices
        L1 = find(tri(:,1)~=0 & tri(:,2)==0 & tri(:,3)==0);
        L2 = find(tri(:,1)~=0 & tri(:,2)~=0 & tri(:,3)==0);
        L3 = find(all(tri~=0,2));

        % PSTHs -----------------------------------------------------------
        E1L1(n,:) = averagePSTH(L1, ts{1}, SU, binSzEnc, encWindow);
        E1L2(n,:) = averagePSTH(L2, ts{1}, SU, binSzEnc, encWindow);
        E1L3(n,:) = averagePSTH(L3, ts{1}, SU, binSzEnc, encWindow);

        E2L2(n,:) = averagePSTH(L2, ts{2}, SU, binSzEnc, encWindow);
        E2L3(n,:) = averagePSTH(L3, ts{2}, SU, binSzEnc, encWindow);

        E3L3(n,:) = averagePSTH(L3, ts{3}, SU, binSzEnc, encWindow);

        % Maintenance -----------------------------------------------------
        if ~isempty(L1)
            M1(n,:) = mean(nd.firing_rates(L1,:),1,'omitnan');
        end
    end

    % ---------- per‑neuron min–max scaling -------------------------------
    mats = {E1L1,E1L2,E1L3,E2L2,E2L3,E3L3,M1};   % cell array for easy loop
    for i = 1:N
        combo = [E1L1(i,:),E1L2(i,:),E1L3(i,:), ...
                 E2L2(i,:),E2L3(i,:),E3L3(i,:),M1(i,:)];
        mn = min(combo,[],'omitnan'); mx = max(combo,[],'omitnan');
        if isnan(mn) || mx <= mn, mn = 0; mx = 1; end

        for k = 1:numel(mats)
            mats{k}(i,:) = (mats{k}(i,:) - mn) / (mx - mn);
        end
    end
    [E1L1,E1L2,E1L3,E2L2,E2L3,E3L3,M1] = mats{:};

    % ---------- sort by Maintenance‑L1 peak latency ----------------------
    [~,pk] = max(M1,[],2,'includenan');
    [~,ord]= sort(pk,'ascend');

    E1L1 = E1L1(ord,:);  E1L2 = E1L2(ord,:);  E1L3 = E1L3(ord,:);
    E2L2 = E2L2(ord,:);  E2L3 = E2L3(ord,:);  E3L3 = E3L3(ord,:);
    M1   = M1(ord,:);

    % ---------- plot -----------------------------------------------------
    % … everything up to the plotting section is identical …

    %% ---------- plot -----------------------------------------------------
    figure('Name','All encoding phases & Maintenance | unified per‑unit scaling');

    xLabs = 0:0.5:encWindow;
    xt    = round(xLabs/binSzEnc)+1;

    % ---- helper nested inside the main function ----
    function heatPanel(mat, titleTxt)
        imagesc(mat);
        colormap('parula');
        caxis([0 1]);
        colorbar;
        title(titleTxt);
        set(gca,'XTick',xt,'XTickLabel', ...
            arrayfun(@(x) sprintf('%.1f',x), xLabs,'UniformOutput',false));
        xlabel('Time (s)');
        ylabel('Neuron (sorted)');
    end

    % first row (Enc1 L1‑L3)
    subplot(3,3,1); heatPanel(E1L1,'Enc1 Load1');
    subplot(3,3,2); heatPanel(E1L2,'Enc1 Load2');
    subplot(3,3,3); heatPanel(E1L3,'Enc1 Load3');

    % second row (Enc2/3 loads)
    subplot(3,3,4); heatPanel(E2L2,'Enc2 Load2');
    subplot(3,3,5); heatPanel(E2L3,'Enc2 Load3');
    subplot(3,3,6); heatPanel(E3L3,'Enc3 Load3');

    % third row (Maintenance L1) – wider panel
    subplot(3,3,[7 8 9]);
    imagesc(M1);
    colormap('parula');
    caxis([0 1]);
    colorbar;
    title('Maintenance Load1');
    xtMaint = round(linspace(1,numBinsMnt,6));
    set(gca,'XTick',xtMaint,'XTickLabel',arrayfun(@num2str,0:0.5:2.5,'UniformOutput',false));
    xlabel('Time (s)');
    ylabel('Neuron (sorted)');

    fprintf('\nSeven heat‑maps plotted with unified min–max per neuron.\n');



    %% ================= EXTRA FIGURE 2 : load‑pooled encoding heat‑maps ======
    % ---- 1) build the three group matrices ----------------------------------
    encL1  = E1L1;                           % Enc1 Load1            (#neurons × 10)
    encL2  = [E1L2  E2L2];                   % Enc1+Enc2 Load2       (#neurons × 20)
    encL3  = [E1L3  E2L3  E3L3];             % Enc1+Enc2+Enc3 Load3  (#neurons × 30)
    
    % ---- 2) per‑neuron min‑max across the three groups ----------------------
    encL1n = nan(size(encL1)); encL2n = nan(size(encL2)); encL3n = nan(size(encL3));
    for i = 1:size(encL1,1)
        pool = [encL1(i,:), encL2(i,:), encL3(i,:)];
        mn = min(pool,[],'omitnan'); mx = max(pool,[],'omitnan');
        if isnan(mn) || mx <= mn, mn = 0; mx = 1; end
        s = mx - mn;
        encL1n(i,:) = (encL1(i,:) - mn) / s;
        encL2n(i,:) = (encL2(i,:) - mn) / s;
        encL3n(i,:) = (encL3(i,:) - mn) / s;
    end
    
    % ---- 3) plot ------------------------------------------------------------
    figure('Name','FIG 2 | load‑pooled encodings (per‑unit min–max across three groups)');
    
    subplot(1,3,1);
    imagesc(encL1n);  colormap('parula'); caxis([0 1]); colorbar;
    title('Load 1  (Enc1)'); xlabel('Time (s)'); ylabel('Neuron');
    xt  = round((0:0.5:encWindow)/binSzEnc)+1;
    set(gca,'XTick',xt,'XTickLabel',arrayfun(@(x) sprintf('%.1f',x),0:0.5:encWindow,'UniformOutput',0));
    
    subplot(1,3,2);
    imagesc(encL2n);  colormap('parula'); caxis([0 1]); colorbar;
    title('Load 2  (Enc1+2)'); xlabel('Time (s)'); ylabel('Neuron');
    xt2 = round((0:0.5:encWindow*2)/binSzEnc)+1;
    set(gca,'XTick',xt2,'XTickLabel',arrayfun(@(x) sprintf('%.1f',x),0:0.5:encWindow*2,'UniformOutput',0));
    
    subplot(1,3,3);
    imagesc(encL3n);  colormap('parula'); caxis([0 1]); colorbar;
    title('Load 3  (Enc1+2+3)'); xlabel('Time (s)'); ylabel('Neuron');
    xt3 = round((0:0.5:encWindow*3)/binSzEnc)+1;
    set(gca,'XTick',xt3,'XTickLabel',arrayfun(@(x) sprintf('%.1f',x),0:0.5:encWindow*3,'UniformOutput',0));
    
    fprintf('FIG 2 created: three load‑pooled heat‑maps with shared per‑unit scaling.\n');
    
    %% ================ EXTRA FIGURE 3 : bar‑plot of six encoding sessions ====
    % ---- 1) mean firing‑rate per neuron, un‑normalised ----------------------
    meanHz = @(mat) mean(mat,2,'omitnan');   % per‑neuron mean
    
    FR_E1L1 = meanHz(E1L1);   FR_E1L2 = meanHz(E1L2);   FR_E1L3 = meanHz(E1L3);
    FR_E2L2 = meanHz(E2L2);   FR_E2L3 = meanHz(E2L3);   FR_E3L3 = meanHz(E3L3);
    
    % assemble into [#neurons × 6] matrix
    FRmat = [FR_E1L1, FR_E1L2, FR_E1L3, FR_E2L2, FR_E2L3, FR_E3L3];
    
    % ---- 2) group means & SEM ----------------------------------------------
    mFR   = mean(FRmat,1,'omitnan');
    semFR = std(FRmat,0,1,'omitnan') ./ sqrt(sum(~isnan(FRmat),1));
    
    % ---- 3) plot ------------------------------------------------------------
    figure('Name','FIG 3 | Mean firing‑rate per encoding session');
    
    bar(1:6, mFR, 'FaceColor',[0.3 0.55 0.8]); hold on;
    errorbar(1:6, mFR, semFR, '.k', 'LineWidth',1.5);
    
    set(gca,'XTick',1:6,'XTickLabel',{'E1‑L1','E1‑L2','E1‑L3','E2‑L2','E2‑L3','E3‑L3'});
    ylabel('Mean FR  (Hz)');
    title('Encoding sessions: population mean ± SEM');
    
    fprintf('FIG 3 created: six‑bar firing‑rate summary.\n');

end   % <-- end of main function
     

% ======================== helper =========================================
function row = averagePSTH(idx, tsEnc, SU, binSz, win)
    if isempty(idx) || isempty(tsEnc)
        row = nan(1,round(win/binSz)); return
    end
    edges = 0:binSz:win;                 % relative edges (row length = win/binSz)
    out   = zeros(numel(idx),numel(edges)-1);
    for j = 1:numel(idx)
        if idx(j) > numel(tsEnc), continue, end
        out(j,:) = histcounts(SU.spike_times, tsEnc(idx(j))+edges) / binSz;
    end
    row = mean(out,1,'omitnan');
end
