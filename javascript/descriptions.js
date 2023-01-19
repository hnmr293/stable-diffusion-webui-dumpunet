onUiUpdate(() => {
    if (!globalThis.DumpUnet) {
        globalThis.DumpUnet = {};
    }
    const DumpUnet = globalThis.DumpUnet;

    DumpUnet.addDescriptionCallback = function () {
        if (DumpUnet.addDescriptionCallbackCalled) return;

        const app = gradioApp();
        if (!app || app === document) return;

        const descs = {
            '#dumpunet-{}-features-checkbox': 'Extract U-Net features and add their maps to output images.',
            '#dumpunet-{}-features-layer': 'U-Net layers <code>(IN00-IN11, M00, OUT00-OUT11)</code> which features should be extracted. See tooltip for notations.',
            '#dumpunet-{}-features-steps': 'Steps which U-Net features should be extracted. See tooltip for notations',
            '#dumpunet-{}-features-dumppath': 'Raw binary files are dumped to here, one image per step per layer.',

            '#dumpunet-{}-layerprompt-checkbox': 'When checked, <code>(~: ... :~)</code> notation is enabled.',
            '#dumpunet-{}-layerprompt-stdout-checkbox': 'When checked, print prompts for each batch to stdout.',
        };

        const hints = {
            '#dumpunet-{}-features-layer textarea': 'IN00: add one layer to output\nIN00,IN01: add layers to output\nIN00-IN02: add range to output\nIN00-OUT05(+2): add range to output with specified steps\n',
            '#dumpunet-{}-features-steps textarea': '5: extracted at steps=5\n5,10: extracted at steps=5 and steps=10\n5-10: extracted when step is in 5..10 (inclusive)\n5-10(+2): extracts when step is 5,7,9\n',
        };

        for (let [k, v] of Object.entries(descs)) {
            const cont = document.createElement('div');
            cont.innerHTML = v;
            cont.classList.add('dumpunet-description');
            for (let x of ['txt2img', 'img2img']) {
                app.querySelector(k.replace('{}', x)).append(cont.cloneNode(true));
            }
        }

        for (let [k, v] of Object.entries(hints)) {
            const cont = document.createElement('pre');
            cont.innerHTML = v;
            cont.classList.add('dumpunet-tooltip');
            for (let x of ['txt2img', 'img2img']) {
                const parent = app.querySelector(k.replace('{}', x)).parentNode;
                parent.classList.add('dumpunet-tooltip-parent');
                parent.append(cont.cloneNode(true));
            }
        }

        DumpUnet.addDescriptionCallbackCalled = true;
    };

    onUiUpdate(DumpUnet.addDescriptionCallback);
});
