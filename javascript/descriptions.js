onUiUpdate(() => {
    if (!globalThis.DumpUnet) {
        globalThis.DumpUnet = {};
    }
    const DumpUnet = globalThis.DumpUnet;

    DumpUnet.addDescriptionCallback = function () {
        if (DumpUnet.addDescriptionCallbackCalled) return;

        const app = gradioApp();
        if (!app || app === document) return;
        if (!app.querySelector('#dumpunet-txt2img-ui')
            && !app.querySelector('#dumpunet-img2img-ui'))
            return;

        const descs = {
            '#dumpunet-{}-features-checkbox': 'Extract U-Net features and add their maps to output images.',
            '#dumpunet-{}-features-layer': 'U-Net layers <code>(IN00-IN11, M00, OUT00-OUT11)</code> which features should be extracted. See tooltip for notations.',
            '#dumpunet-{}-features-steps': 'Steps which U-Net features should be extracted. See tooltip for notations.',
            '#dumpunet-{}-features-average': 'Add channel-averaged map to the result.',
            '#dumpunet-{}-features-dumppath': 'Raw binary files are dumped to here, one image per step per layer.',
            '#dumpunet-{}-features-colorization-desc': 'Recommends for U-Net features: <code>Custom / Sigmoid (gain=1.0, offset=0.0) / HSL; H=(2+v)/3, S=1.0, L=0.5</code>',
            '#dumpunet-{}-features-colorization-custom': 'Set RGB/HSL value with given transformed value <code>v</code>. The range of <code>v</code> can be either [0, 1] or [-1, 1] according to the `Value transform` selection.<br/>Input values are processed as `eval(f"lambda v: ( ({r}), ({g}), ({b}) )", { "__builtins__": numpy }, {})`.',

            '#dumpunet-{}-attention-checkbox': 'Extract attention layer\'s features and add their maps to output images.',
            '#dumpunet-{}-attention-layer': 'U-Net layers <code>(IN00-IN11, M00, OUT00-OUT11)</code> which features should be extracted. See tooltip for notations.',
            '#dumpunet-{}-attention-steps': 'Steps which features should be extracted. See tooltip for notations.',
            '#dumpunet-{}-attention-average': 'For K, add head-averaged map.<br/>For Q*K, add head-averaged map.<br/>For V*Q*K, add channel-averaged map.',
            '#dumpunet-{}-attention-dumppath': 'Raw binary files are dumped to here, one image per step per layer.',
            '#dumpunet-{}-attention-colorization-desc': 'Recommends for Attention features: <code>Custom / Auto [0,1] / HSL; H=(2-2*v)/3, S=1.0, L=0.5</code>',
            '#dumpunet-{}-attention-colorization-custom': 'Set RGB/HSL value with given transformed value <code>v</code>. The range of <code>v</code> can be either [0, 1] or [-1, 1] according to the `Value transform` selection.<br/>Input values are processed as `eval(f"lambda v: ( ({r}), ({g}), ({b}) )", { "__builtins__": numpy }, {})`.',

            '#dumpunet-{}-layerprompt-checkbox': 'When checked, <code>(~: ... :~)</code> notation is enabled.',
            '#dumpunet-{}-layerprompt-diff-layer': 'Layers <code>(IN00-IN11, M00, OUT00-OUT11)</code> which features should be extracted. See tooltip for notations.',
            '#dumpunet-{}-layerprompt-diff-steps': 'Steps which features should be extracted. See tooltip for notations.',
            '#dumpunet-{}-layerprompt-diff-average': 'Add channel-averaged map to the result.',
            '#dumpunet-{}-layerprompt-diff-dumppath': 'Raw binary files are dumped to here, one image per step per layer.',
            '#dumpunet-{}-layerprompt-diff-colorization-desc': 'Recommends for layer prompt\'s diff.: <code>Custom / Sigmoid (gain=1.0, offset=0.0) / HSL; H=(2+v)/3, S=1.0, L=0.5</code>',
            '#dumpunet-{}-layerprompt-diff-colorization-custom': 'Set RGB/HSL value with given transformed value <code>v</code>. The range of <code>v</code> can be either [0, 1] or [-1, 1] according to the `Value transform` selection.<br/>Input values are processed as `eval(f"lambda v: ( ({r}), ({g}), ({b}) )", { "__builtins__": numpy }, {})`.',
        };

        const hints = {
            '#dumpunet-{}-features-layer textarea': 'IN00: add one layer to output\nIN00,IN01: add layers to output\nIN00-IN02: add range to output\nIN00-OUT05(+2): add range to output with specified steps\n',
            '#dumpunet-{}-features-steps textarea': '5: extracted at steps=5\n5,10: extracted at steps=5 and steps=10\n5-10: extracted when step is in 5..10 (inclusive)\n5-10(+2): extracts when step is 5,7,9\n',
            '#dumpunet-{}-features-colorization-method label:nth-child(1) > *:first-child': 'Grayscale output. |v|=1 is white, |v|=0 is black.',
            '#dumpunet-{}-features-colorization-method label:nth-child(2) > *:first-child': 'Red/Blue output. v=1 is red, v=-1 is blue.',
            '#dumpunet-{}-features-colorization-method label:nth-child(3) > *:first-child': 'Custom output. Specify color via <code>Color space</code> area below.',
            '#dumpunet-{}-features-colorization-trans label:nth-child(1) > *:first-child': 'Auto [0,1]: linearly transform values to [0, 1].',
            '#dumpunet-{}-features-colorization-trans label:nth-child(2) > *:first-child': 'Auto [-1,1]: linearly transform values to [-1, 1].',
            '#dumpunet-{}-features-colorization-trans label:nth-child(3) > *:first-child': 'Linear: linearly transform values from [Clamp min., Clamp max.] to [0, 1] (for White/Black mode) or [-1, 1] (otherwise).',
            '#dumpunet-{}-features-colorization-trans label:nth-child(4) > *:first-child': 'Sigmoid: transform values from [-inf., +inf.] to [0, 1] (for White/Black mode) or [-1, 1] (otherwise).',

            '#dumpunet-{}-attention-layer textarea': 'IN00: add one layer to output\nIN00,IN01: add layers to output\nIN00-IN02: add range to output\nIN00-OUT05(+2): add range to output with specified steps\n',
            '#dumpunet-{}-attention-steps textarea': '5: extracted at steps=5\n5,10: extracted at steps=5 and steps=10\n5-10: extracted when step is in 5..10 (inclusive)\n5-10(+2): extracts when step is 5,7,9\n',
            '#dumpunet-{}-attention-colorization-method label:nth-child(1) > *:first-child': 'Grayscale output. |v|=1 is white, |v|=0 is black.',
            '#dumpunet-{}-attention-colorization-method label:nth-child(2) > *:first-child': 'Red/Blue output. v=1 is red, v=-1 is blue.',
            '#dumpunet-{}-attention-colorization-method label:nth-child(3) > *:first-child': 'Custom output. Specify color via <code>Color space</code> area below.',
            '#dumpunet-{}-attention-colorization-trans label:nth-child(1) > *:first-child': 'Auto [0,1]: linearly transform values to [0, 1].',
            '#dumpunet-{}-attention-colorization-trans label:nth-child(2) > *:first-child': 'Auto [-1,1]: linearly transform values to [-1, 1].',
            '#dumpunet-{}-attention-colorization-trans label:nth-child(3) > *:first-child': 'Linear: linearly transform values from [Clamp min., Clamp max.] to [0, 1] (for White/Black mode) or [-1, 1] (otherwise).',
            '#dumpunet-{}-attention-colorization-trans label:nth-child(4) > *:first-child': 'Sigmoid: transform values from [-inf., +inf.] to [0, 1] (for White/Black mode) or [-1, 1] (otherwise).',

            '#dumpunet-{}-layerprompt-diff-layer textarea': 'IN00: add one layer to output\nIN00,IN01: add layers to output\nIN00-IN02: add range to output\nIN00-OUT05(+2): add range to output with specified steps\n',
            '#dumpunet-{}-layerprompt-diff-steps textarea': '5: extracted at steps=5\n5,10: extracted at steps=5 and steps=10\n5-10: extracted when step is in 5..10 (inclusive)\n5-10(+2): extracts when step is 5,7,9\n',
            '#dumpunet-{}-layerprompt-diff-colorization-method label:nth-child(1) > *:first-child': 'Grayscale output. |v|=1 is white, |v|=0 is black.',
            '#dumpunet-{}-layerprompt-diff-colorization-method label:nth-child(2) > *:first-child': 'Red/Blue output. v=1 is red, v=-1 is blue.',
            '#dumpunet-{}-layerprompt-diff-colorization-method label:nth-child(3) > *:first-child': 'Custom output. Specify color via <code>Color space</code> area below.',
            '#dumpunet-{}-layerprompt-diff-colorization-trans label:nth-child(1) > *:first-child': 'Auto [0,1]: linearly transform values to [0, 1].',
            '#dumpunet-{}-layerprompt-diff-colorization-trans label:nth-child(2) > *:first-child': 'Auto [-1,1]: linearly transform values to [-1, 1].',
            '#dumpunet-{}-layerprompt-diff-colorization-trans label:nth-child(3) > *:first-child': 'Linear: linearly transform values from [Clamp min., Clamp max.] to [0, 1] (for White/Black mode) or [-1, 1] (otherwise).',
            '#dumpunet-{}-layerprompt-diff-colorization-trans label:nth-child(4) > *:first-child': 'Sigmoid: transform values from [-inf., +inf.] to [0, 1] (for White/Black mode) or [-1, 1] (otherwise).',
        };

        for (let [k, v] of Object.entries(descs)) {
            const cont = document.createElement('div');
            cont.innerHTML = v;
            cont.classList.add('dumpunet-description');
            for (let x of ['txt2img', 'img2img']) {
                const q = k.replace('{}', x);
                const ele = app.querySelector(q);
                if (!ele) {
                    console.warn(`"${q}" not found`);
                    continue;
                }
                ele.append(cont.cloneNode(true));
            }
        }

        for (let [k, v] of Object.entries(hints)) {
            const cont = document.createElement('pre');
            cont.innerHTML = v;
            cont.classList.add('dumpunet-tooltip');
            for (let x of ['txt2img', 'img2img']) {
                const q = k.replace('{}', x);
                const ele = app.querySelector(q);
                if (!ele) {
                    console.warn(`"${q}" not found`);
                    continue;
                }
                const parent = ele.parentNode;
                parent.classList.add('dumpunet-tooltip-parent');
                parent.append(cont.cloneNode(true));
            }
        }

        DumpUnet.addDescriptionCallbackCalled = true;
    };

    onUiUpdate(DumpUnet.addDescriptionCallback);
});
