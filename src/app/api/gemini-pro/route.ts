// api/gemini/route.ts
import { GoogleGenerativeAIStream, Message, StreamingTextResponse } from "ai";

import {
  GoogleGenerativeAI,
  GenerateContentRequest,
  Content,
} from "@google/generative-ai";

import {
  mapSafetySettings,
  defaultSafetySettings,
} from "@/lib/safety-settings-mapper";

import { sanitizeContent } from "@/lib/sanitize-content";

import { proRequestSchema } from "@/lib/validate/pro-request-schema";

import { GeneralSettings } from "@/types";

function createOnceFunction() {
  let hasRun = false;

  return function() {
    const executed = !hasRun; // Flag to indicate execution
    if (!hasRun) {
      console.log("Function executed only once!");
      hasRun = true;
    }
    return executed; // Return execution status
  };
}

export const runtime = "edge";

export async function POST(req: Request) {
  const myOnceFunction = createOnceFunction();
  const parseResult = proRequestSchema.safeParse(await req.json());
  var initialText = "Pretend you are Utkarsh Rai, a medical images researcher at University of Arkansas for Medical Sciences. Your core research is in HTJ2K compression of DICOM images and LLM generated Radiology Reports. Previously you have worked as a data engineer with two great companies and have a wide range of skills. Introduce yourself. And then respond to : ";
  if (!parseResult.success) {
    // If validation fails, return a 400 Bad Request response
    return new Response(JSON.stringify({ error: "Invalid request data" }), {
      status: 400,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  const { messages, general_settings, safety_settings } = parseResult.data;
  const { temperature, maxLength, topP, topK } =
    general_settings as GeneralSettings;

  for (const message of messages) {
    message.content = sanitizeContent(message.content);
  }

  const typedMessages: Message[] = messages as unknown as Message[];

  // consecutive user messages need to be merged into the same content, 2 consecutive Content objects with user role will error with the Gemini api
  var reqContent: GenerateContentRequest = {
    contents: typedMessages.reduce((acc: Content[], m: Message) => {
      if (m.role === "user") {
        const lastContent = acc[acc.length - 1];
        if (lastContent && lastContent.role === "user") {
            if (myOnceFunction()){
              lastContent.parts.push({ text: initialText+m.content });
            }
            else{
              lastContent.parts.push({ text: m.content });
            }
          
        } else {
            if (myOnceFunction()){
              acc.push({
                role: "user",
                parts: [{ text:initialText + m.content }],
              });
            }
            else{
              acc.push({
                role: "user",
                parts: [{ text: m.content }],
              });
            }
          
        }
      } else if (m.role === "assistant") {
        acc.push({
          role: "model",
          parts: [{ text: m.content }],
        });
      }

      return acc;
    }, []),
  };

  const incomingSafetySettings = safety_settings || defaultSafetySettings;
  const mappedSafetySettings = mapSafetySettings(incomingSafetySettings);

  const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY as string);

  const tokens = await genAI
    .getGenerativeModel({
      model: "gemini-pro",
    })
    .countTokens(reqContent);
  console.log("count tokens ------", tokens);
  
  
  
  const geminiStream = await genAI
    .getGenerativeModel({
      model: "gemini-pro",
      safetySettings: mappedSafetySettings,
      generationConfig: {
        //   candidateCount: 0,
        //   stopSequences: [],
        maxOutputTokens: maxLength,
        temperature,
        topP,
        topK,
      },
    })
    .generateContentStream(reqContent);

  const stream = GoogleGenerativeAIStream(geminiStream);

  return new StreamingTextResponse(stream);
}
