// api/gemini/route.ts
import { Message } from "ai";

import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
  Content,
} from "@google/generative-ai";

const safetySettings = [
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
];

export const runtime = "edge";

export async function POST(req: Request) {
  const { messages, temperature, max_length, top_p, top_k } = await req.json();
  // console.log(temperature, max_length, top_p, top_k);

  const chatHistory: Content[] = messages.map((m: Message) => {
    if (m.role === "user") {
      return {
        role: "user",
        parts: [{ text: m.content }],
      };
    }
    if (m.role === "assistant") {
      return {
        role: "model",
        parts: [{ text: m.content }],
      };
    }
    return undefined;
  });

  const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY as string);
  const model = genAI.getGenerativeModel({
    model: "gemini-pro",
    safetySettings,
    generationConfig: {
      //   candidateCount: 0,
      //   stopSequences: [],
      maxOutputTokens: max_length,
      temperature: temperature,
      topP: top_p,
      topK: top_k,
    },
  });

  try {
    const streamingResp = await model.generateContentStream({
      contents: chatHistory,
    });

    const stream = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of streamingResp.stream) {
            if (chunk.candidates) {
              const parts = chunk.candidates[0].content.parts;
              const firstPart = parts[0];

              if (typeof firstPart.text === "string") {
                controller.enqueue(firstPart.text);
              }
            }
          }
          controller.close();
        } catch (error) {
          console.error("Streaming error:", error);
          controller.error(error);
        }
      },
    });

    return new Response(stream, {
      headers: { "Content-Type": "text/plain" },
    });
  } catch (error) {
    console.error("API error:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}
