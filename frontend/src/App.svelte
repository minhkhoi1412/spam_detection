<script>
    let data = null;
    let message = "";

    const predict = async () => {
        const response = await fetch(
            "http://localhost:5000/?data=" + encodeURIComponent(message),
            {
                method: "POST",
                headers: {
                    Accept: "application/json",
                    "Content-Type": "application/json",
                },
            }
        );

        data = await response.json();
        console.log(data);
    };

    const sendMessage = () => {
        predict(message);
    };
</script>

<main>
    <div class="message-box">
        <textarea
            bind:value={message}
            name="Message"
            class="message"
            id="message"
            cols="30"
            rows="10"
        />
        <button class="message-button" on:click={sendMessage}>Send</button>
        <div class="message-predict">
            {#if data}
                {#if data["prediction"] === 0}
                    <p>Đây là tin nhắn ham</p>
                {:else}
                    <p>Đây là tin nhắn spam</p>
                {/if}
            {/if}
        </div>
    </div>
</main>

<style>
    main {
        position: relative;
        width: 100%;
        height: 100%;
    }

    .message-box {
        position: absolute;
        width: 350px;
        height: 500px;
        bottom: 30px;
        right: 30px;
        border: 1px solid #bbb;
    }

    .message {
        width: 100%;
        height: 350px;
        margin: 0;
        padding: 10px;
        border: 0;
        outline: none;
        resize: none;
        line-height: 25px;
    }

    .message-button {
        width: 100%;
        height: 50px;
        background-color: #eee;
        margin: 0;
        border-left: 0;
        border-right: 0;
    }

    .message-predict {
        width: 100%;
        height: 100px;
        padding: 10px;
        line-height: 25px;
    }
</style>
